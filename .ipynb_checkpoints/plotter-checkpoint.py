import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import struct
import re
from abc import ABC, abstractmethod
from numba import njit
from concurrent.futures import ThreadPoolExecutor

import yaml
import matplotlib.ticker as mticker
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Angle

try:
    from tqdm import tqdm
except Exception:
    tqdm = None
    
# Set matplotlib style
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1),
})




class Station:
    """A class to hold information about a single antenna station."""
    def __init__(self, station_id, name, lat, lon, elev):
        self.id = int(station_id)
        self.name = str(name)
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.elevation = float(elev) if elev is not None else 0.0
        self.x, self.y, self.z = None, None, None

    def __repr__(self):
        return f"Station(id={self.id}, name='{self.name}')"

class DSA110Array:
    """Manages an array of antenna stations, computes phases and phase matrix."""
    def __init__(self, yaml_path):
        self.path = yaml_path
        self.stations = {}    # Key: station_id, Value: Station
        self.flagged = set()  # Set of flagged station IDs
        self._load_yaml()
        self.to_itrs()

    def _load_yaml(self):
        """Loads and parses the station coordinate data from a YAML file."""
        with open(self.path, 'r') as f:
            raw_data = yaml.safe_load(f)
        self.stations = {}
        for idx, values in raw_data.items():
            idx_int = int(idx)
            self.stations[idx_int] = Station(
                station_id=idx_int,
                name=values.get('Station'),
                lat=values.get('Latitude', 0.0),
                lon=values.get('Longitude', 0.0),
                elev=values.get('Elevation')
            )

    def to_itrs(self):
        """Converts geodetic (Lat, Lon, Elevation) to Earth-centered ITRS (X, Y, Z) for all stations."""
        for s in self.stations.values():
            loc = EarthLocation.from_geodetic(
                lon=s.longitude * u.deg,
                lat=s.latitude * u.deg,
                height=s.elevation * u.m
            )
            s.x, s.y, s.z = loc.x.value, loc.y.value, loc.z.value

    def get_station_ids(self):
        """Return a numpy array of station ids sorted."""
        return np.array(sorted(self.stations))

    def get_station_locations(self):
        """Return Nx3 array of ITRS (X,Y,Z) for all stations, sorted by station ID."""
        ids = self.get_station_ids()
        return np.array([[self.stations[i].x, self.stations[i].y, self.stations[i].z] for i in ids])

    def get_station_by_id(self, stid):
        """Returns the Station object for a given station id."""
        return self.stations[stid]

    def source_vector_ecef(self, ra_deg, dec_deg, obstime):
        """
        Given RA/Dec in deg and astropy Time, returns ECEF unit vector for the source.
        """
        lats = [s.latitude for s in self.stations.values()]
        lons = [s.longitude for s in self.stations.values()]
        elevs = [s.elevation for s in self.stations.values()]
        mean_lat = np.mean(lats)
        mean_lon = np.mean(lons)
        mean_elev = np.mean(elevs)
        loc = EarthLocation.from_geodetic(mean_lon*u.deg, mean_lat*u.deg, mean_elev*u.m)

        skyc = SkyCoord(
            ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs', obstime=obstime, location=loc
        )
        itrs = skyc.transform_to('itrs')
        d_xyz = np.array([itrs.cartesian.x.value, itrs.cartesian.y.value, itrs.cartesian.z.value])
        norm = np.linalg.norm(d_xyz)
        return d_xyz / norm

    def geometric_delay(self, source_unit_vector, ref_index=0):
        """
        Returns array of delays [sec] for all stations given source unit vector (ECEF),
        relative to the ref_index-th station (ordered by station id).
        """
        ids = self.get_station_ids()
        locs = self.get_station_locations()
        ref = locs[ref_index]
        delays = np.dot(locs - ref, source_unit_vector) / 299792458.0  # c in m/s
        return delays  # shape (N_ant,)

    def per_antenna_phase(self, ra_deg, dec_deg, obstime, frequencies_hz, ref_index=0):
        """
        Frequencies can be a scalar or 1D array in Hz.
        Returns phase (N_freq, N_ant) in radians for multi-freq,
        or (N_ant,) for single freq.
        """
        freqs = np.atleast_1d(frequencies_hz).astype(float)  # (N_freq,)
        svec = self.source_vector_ecef(ra_deg, dec_deg, obstime)
        delays = self.geometric_delay(svec, ref_index=ref_index)  # shape (N_ant,)
        phases = -2.0 * np.pi * freqs[:, None] * delays[None, :]
        if phases.shape[0] == 1:
            phases = phases[0]
        return phases

    def phase_matrix(self, ra_deg, dec_deg, obstime, frequencies_hz, ref_index=0):
        """
        Frequencies can be a scalar or 1D array in Hz.
        Returns phase matrix of shape [N_freq, N_ant, N_ant] with
        exp(1j*(phase_i - phase_j)) for each frequency.
        If only one frequency, returns [N_ant, N_ant].
        """
        phases = self.per_antenna_phase(ra_deg, dec_deg, obstime, frequencies_hz, ref_index=ref_index)
        if phases.ndim == 1:
            phase_diff = phases[:, None] - phases[None, :]
            return np.exp(1j * phase_diff)
        else:
            phase_diff = phases[:, :, None] - phases[:, None, :]  # (N_freq, N_ant, N_ant)
            return np.exp(1j * phase_diff)

    def flagger(self, indices):
        """Flags specified stations by adding their indices to a set."""
        for idx in indices:
            if idx in self.stations:
                self.flagged.add(idx)
            else:
                print(f"Warning: Index {idx} not found in list of stations!")

    def plot_positions(self, frame='geo'):
        """
        Plots station positions.
        frame : str
            'geo'  = plot Longitude/Latitude (default)
            'enu'  = plot (East, North) in meters, origin=ref antenna (lowest id)
        """
        ids = self.get_station_ids()
        flagged = set(getattr(self, 'flagged', set()))

        if frame == 'geo':
            xvals = [self.stations[i].longitude for i in ids]
            yvals = [self.stations[i].latitude for i in ids]
            xlab = 'Longitude (deg)'
            ylab = 'Latitude (deg)'
            core_mask = np.array([i < 103 for i in ids])
        elif frame == 'enu':
            ref_s = self.stations[ids[0]]
            ref_loc = EarthLocation.from_geodetic(ref_s.longitude*u.deg, ref_s.latitude*u.deg, ref_s.elevation*u.m)
            xyz_ref = np.array([ref_s.x, ref_s.y, ref_s.z])
            lon_rad = np.deg2rad(ref_s.longitude)
            lat_rad = np.deg2rad(ref_s.latitude)
            s_lon, c_lon = np.sin(lon_rad), np.cos(lon_rad)
            s_lat, c_lat = np.sin(lat_rad), np.cos(lat_rad)
            R = np.array([
                [-s_lon,          c_lon,           0],
                [-s_lat*c_lon, -s_lat*s_lon,   c_lat],
                [ c_lat*c_lon,  c_lat*s_lon,   s_lat]
            ])
            xvals, yvals, core_mask = [], [], []
            for s in (self.stations[i] for i in ids):
                xyz = np.array([s.x, s.y, s.z])
                enu = R @ (xyz - xyz_ref)
                xvals.append(enu[0])
                yvals.append(enu[1])
                core_mask.append(s.id < 103)
            xlab = 'East (m)'
            ylab = 'North (m)'
            core_mask = np.array(core_mask)
        else:
            raise ValueError("frame must be 'geo' or 'enu'")

        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax1.set_title("Full Array")
        for idx, x, y in zip(ids, xvals, yvals):
            color = "red" if idx in flagged else "black"
            ax1.plot(x, y, '.', color=color)
            if idx >= 103:
                ax1.text(x, y, str(idx), color=color, fontsize=6, ha='right', rotation=45)

        ax2.set_title("Core Array")
        for idx, x, y, core in zip(ids, xvals, yvals, core_mask):
            if core:
                color = "red" if idx in flagged else "black"
                ax2.plot(x, y, '.', color=color)
                ax2.text(x, y, str(idx), color=color, fontsize=6, ha='right', va='bottom', rotation=45)

        for ax in [ax1, ax2]:
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box')
            if frame == 'geo':
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            else:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))

        plt.show()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def plot_waterfall(data,times,frequencies, plotter, figsize=(12, 12)):
        """Plot waterfall for given antenna showing power spectra for X, Y polarizations and Stokes I (optimized)"""
        # Load data on demand
        combined_data = data.copy()
        times = times.copy()

        pol_x = combined_data['pol_x']
        pol_y = combined_data['pol_y']
        pol_x_power = pol_x.real**2 + pol_x.imag**2
        pol_y_power = pol_y.real**2 + pol_y.imag**2

        # Calculate Stokes I
        stokes_i = pol_x_power + pol_y_power

        ch_median = np.median(stokes_i, axis=0)
        stokes_i -= ch_median
        # Calculate marginal profiles for Stokes I (like DSA110WaterfallPlotter)
        freq_profile = np.mean(stokes_i, axis=0)
        time_profile = np.mean(stokes_i, axis=1)
        snr, _, _ = calculate_snr_(time_profile)

        # Create plot with gridspec layout (matching DSA110WaterfallPlotter style)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05,
                              height_ratios=[1, 3, 3, 3], width_ratios=[3, 3, 3, 1])

        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_time = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_freq = fig.add_subplot(gs[1:, -1], sharey=ax_main)

        # Color scaling (use Stokes I for common scale)
        vmin = np.percentile(stokes_i, 1)
        vmax = np.percentile(stokes_i, 99.9)

        # Main waterfall (Stokes I only, like DSA110WaterfallPlotter)
        im = ax_main.imshow(stokes_i.T, aspect='auto', origin='lower',
                           extent=[times[0], times[-1], frequencies[-1], frequencies[0]],
                           cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")

        ax_main.set_xlabel('Time (seconds)')
        ax_main.set_ylabel('Frequency (MHz)')

        # Flag status and title formatting
        max_snr_text = f", max SNR = {np.max(snr):.1f}"
        fig.suptitle(f'DSA110 Beamformed Stokes I\n'
                     f'{len(frequencies)} channels, {len(times)} time samples{max_snr_text}')

        # Time profile (SNR plot like DSA110WaterfallPlotter)
        ax_time.step(times, snr, where='mid', linewidth=1)
        ax_time.set_ylabel('SNR')
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(labelbottom=False)

        ax_freq.step(freq_profile[::-1], frequencies, where='mid', linewidth=1)
        ax_freq.set_xlabel('arb')
        ax_freq.yaxis.set_label_position('right')
        ax_freq.yaxis.tick_right()
        ax_freq.grid(True, alpha=0.3)
        ax_freq.tick_params(labelleft=False)

        # Colorbar (matching DSA110WaterfallPlotter style)
        cbar = fig.colorbar(im, ax=[ax_main, ax_time, ax_freq],
                           orientation='horizontal', pad=0.1, aspect=60, fraction=0.04)
        cbar.set_label('Stokes I Power (arb. units)')

        plt.show()

        return snr, stokes_i
    
def calculate_snr(
    data,
    rough_snr_thresh=5,
    pulse_snr_thresh=3,
    min_width=1,
    max_pulses=None
):
    """
    Find pulse-like peaks in time series, exclude them from std/MAD calculation,
    and return final SNR, background mean/std, and pulse region indices.

    Parameters
    ----------
    data : np.ndarray
        The input 1D time series data.
    rough_snr_thresh : float
        Initial SNR threshold for detecting seed pulse peaks.
    pulse_snr_thresh : float
        Lower SNR threshold for extending peak region boundaries ("island").
    min_width : int
        Minimum width (# samples) for a valid pulse region.
    max_pulses : int or None
        Maximum number of peaks to report (None = unlimited).

    Returns
    -------
    snr : np.ndarray
        Final SNR using background-only MAD (or std).
    median_val : float
        Background median.
    std : float
        Background std (from MAD).
    pulses : list of (start, stop) tuples
        List of (start_idx, stop_idx) pairs for detected pulse regions.
    """

    # 1. Initial MAD estimate for whole array
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val))
    std = 1.4826 * mad if mad > 0 else np.std(data) or 1.0 # guard against zeros

    rough_snr = (data - median_val) / std

    # 2. Find candidate pulse peaks above rough_snr_thresh
    peak_indices = np.where(rough_snr > rough_snr_thresh)[0]
    if len(peak_indices) == 0:
        # No "strong" peaks - use original full-array MAD
        return rough_snr, median_val, std, []

    # 3. Define pulse regions by expanding above SNR > pulse_snr_thresh
    mask = np.zeros(data.shape, dtype=bool)
    pulses = []
    used = np.zeros_like(data, dtype=bool)
    for idx in peak_indices:
        if used[idx]:
            # skip indices already assigned to a region
            continue

        # Expand left
        left_idx = idx
        while left_idx > 0 and rough_snr[left_idx-1] > pulse_snr_thresh:
            left_idx -= 1
        # Expand right
        right_idx = idx
        while right_idx < len(data)-1 and rough_snr[right_idx+1] > pulse_snr_thresh:
            right_idx += 1
        width = right_idx - left_idx + 1
        if width >= min_width:
            pulses.append((left_idx, right_idx))
            used[left_idx:right_idx+1] = True
            mask[left_idx:right_idx+1] = True

            if max_pulses is not None and len(pulses) >= max_pulses:
                break
    if len(pulses) == 0:
        # No contiguous pulse regions - default to global MAD
        return rough_snr, median_val, std, []
    
    # 4. Mask out pulse regions, calculate MAD and std on background only
    background = data[~mask]
    if len(background) < 4:  # too little background, revert to global
        snr = (data - median_val) / std
        return snr, median_val, std, pulses

    new_median_val = np.median(background)
    new_mad = np.median(np.abs(background - new_median_val))
    new_std = 1.4826 * new_mad if new_mad > 0 else np.std(background) or std
    snr = (data - new_median_val) / new_std
    return snr, new_median_val, new_std, pulses

def calculate_snr_(data):
    """Calculate Signal-to-Noise Ratio using Median Absolute Deviation."""
    median_val = np.median(data)
    absolute_deviations = np.abs(data - median_val)
    mad = np.median(absolute_deviations)
    std = 1.4826 * mad
    
    if std == 0:
        snr = np.zeros_like(data)
    else:
        snr = (data - median_val) / std
        
    return snr, median_val, std

def integrate_data_arrays(data, times, int_time_samples, start_idx=0, stop_idx=-1):
    """Integrate data over time samples."""
    k = int_time_samples
    data_slice = data[start_idx:stop_idx]
    times_slice = times[start_idx:stop_idx]
    
    # Trim to make divisible by k
    n_samples = len(data_slice) // k * k
    data_integrated = data_slice[:n_samples].reshape(-1, k).mean(-1)
    times_integrated = times_slice[:n_samples].reshape(-1, k).mean(-1)
    
    return data_integrated, times_integrated

def time_to_indices_converter(start_time=None, stop_time=None, tsamp=None):
    """Convert time values in seconds to array indices."""
    if tsamp is None:
        raise ValueError("tsamp must be provided")
        
    start_idx = 0 if start_time is None else int(start_time / tsamp)
    stop_idx = -1 if stop_time is None else int(stop_time / tsamp)
    
    return start_idx, stop_idx

def calculate_stokes_parameters(data_x, data_y):
    """Calculate Stokes parameters from X and Y polarization data."""
    stokes_i = np.abs(data_x)**2 + np.abs(data_y)**2
    stokes_q = np.abs(data_x)**2 - np.abs(data_y)**2
    stokes_u = 2 * np.real(data_x * np.conj(data_y))
    stokes_v = 2 * np.imag(data_x * np.conj(data_y))
    
    return stokes_i, stokes_q, stokes_u, stokes_v

@njit(fastmath=True, cache=True)
def calculate_stokes_i(pol_x, pol_y, freq_bin, time_bin):
    """
    Compute Stokes I power and bin in frequency and time using Numba.

    Parameters
    ----------
    pol_x : np.ndarray
        Complex voltage array (time, freq)
    pol_y : np.ndarray or None
        Complex voltage array (time, freq) or None
    freq_bin : int
        Frequency binning factor
    time_bin : int
        Time binning factor

    Returns
    -------
    stokes_i : np.ndarray
        Binned Stokes I power (time, freq)
    """
    tx, fx = pol_x.shape
    t_binned = (tx // time_bin) * time_bin
    f_binned = (fx // freq_bin) * freq_bin
    out_t = t_binned // time_bin
    out_f = f_binned // freq_bin

    stokes_i = np.zeros((out_t, out_f), dtype=np.float32)
    for ti in range(out_t):
        for fi in range(out_f):
            px = pol_x[ti*time_bin:(ti+1)*time_bin, fi*freq_bin:(fi+1)*freq_bin]
            power = np.abs(px)**2
            if pol_y is not None:
                py = pol_y[ti*time_bin:(ti+1)*time_bin, fi*freq_bin:(fi+1)*freq_bin]
                power += np.abs(py)**2
            stokes_i[ti, fi] = np.sum(power)
    return stokes_i

def calculate_polarization_properties(stokes_i, stokes_q, stokes_u, stokes_v):
    """Calculate polarization angle and degree from Stokes parameters."""
    pol_angle = 0.5 * np.arctan2(stokes_u, stokes_q)
    linear_pol = np.sqrt(stokes_q**2 + stokes_u**2) / stokes_i
    circular_pol = np.abs(stokes_v) / stokes_i
    total_pol = np.sqrt(stokes_q**2 + stokes_u**2 + stokes_v**2) / stokes_i
    
    return pol_angle, linear_pol, circular_pol, total_pol


def create_subplot_grid(num_plots, max_cols=4):
    """Create optimal subplot grid for given number of plots."""
    if num_plots <= max_cols:
        return 1, num_plots
    else:
        rows = (num_plots + max_cols - 1) // max_cols
        return rows, max_cols


def _console_progress(current, total, prefix='', bar_length=40):
    """Simple console progress bar (inline)."""
    if total <= 0:
        return
    frac = float(current) / float(total)
    filled = int(round(bar_length * frac))
    bar = '#' * filled + '-' * (bar_length - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({frac*100:5.1f}%)", end='', flush=True)
    if current >= total:
        print()


def _maybe_tqdm(total, desc=None):
    """Return a tqdm progress bar if available, otherwise None."""
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, ncols= 100)

# ============================================================================
# N2 MATRIX CLASS FOR EIGENVALUE ANALYSIS
# ============================================================================

class N2Matrix:
    """
    Class for handling N2 visibility matrices with eigenvalue decomposition and analysis.
    
    The N2 matrix represents the cross-correlation matrix of voltage streams:
    N2[i,j] = V_i * conj(V_j) where V_i and V_j are voltage streams from antennas i and j.
    """
    
    def __init__(self, n2_data, antennas, frequencies, freq_channels, 
                 polarization, sum_frequencies, times=None, metadata=None):
        """
        Initialize N2Matrix object.
        
        Parameters
        ----------
        n2_data : np.ndarray or dict
            N2 matrix data. If dict, keys are frequency channels.
        antennas : list
            List of antenna numbers
        frequencies : list
            List of frequencies in MHz
        freq_channels : list
            List of frequency channel indices
        polarization : str
            Polarization used ('x' or 'y')
        sum_frequencies : bool
            Whether frequencies were summed
        times : np.ndarray, optional
            Time array for the data
        metadata : dict, optional
            Additional metadata
        """
        self.n2_data = n2_data
        self.antennas = antennas
        self.frequencies = frequencies
        self.freq_channels = freq_channels
        self.polarization = polarization
        self.sum_frequencies = sum_frequencies
        self.times = times
        self.metadata = metadata or {}
        
        # Properties derived from data
        self.num_antennas = len(antennas)
        self.num_frequencies = len(frequencies)
        
        if isinstance(n2_data, np.ndarray):
            self.shape = n2_data.shape
            if len(n2_data.shape) == 3:
                # Time-summed case: (num_freq_channels, num_antennas, num_antennas)
                self.num_time_samples = None
                self.time_summed = True
            elif len(n2_data.shape) == 4:
                # Time-resolved case: (num_time_samples, num_freq_channels, num_antennas, num_antennas)
                self.num_time_samples = n2_data.shape[0]
                self.time_summed = False
            else:
                raise ValueError(f"Unexpected N2 matrix shape: {n2_data.shape}")
        else:
            # Dict of matrices (legacy support)
            first_matrix = list(n2_data.values())[0]
            self.shape = f"dict with {len(n2_data)} matrices of shape {first_matrix.shape}"
            self.num_time_samples = first_matrix.shape[0] if len(first_matrix.shape) > 2 else None
            self.time_summed = len(first_matrix.shape) == 2
        
        # Storage for eigenvalue decomposition results
        self.eigenvalues = None
        self.eigenvectors = None
        self.eigendecomposition_done = False
        self.eigendecomposition_metadata = {}
    
    def compute_eigendecomposition(self, time_average=False, freq_channels=None):
        """
        Compute eigenvalue decomposition of the N2 matrix.
        
        Parameters
        ----------
        time_average : bool
            If True, average over time before decomposition.
            If False, compute eigendecomposition for each time sample.
        freq_channels : list or None
            If sum_frequencies=False, specify which frequency channels to decompose.
            If None and sum_frequencies=False, decomposes all channels.
            
        Returns
        -------
        dict
            Dictionary with eigenvalues and eigenvectors
        """
        print("Computing eigenvalue decomposition...")
        
        if isinstance(self.n2_data, np.ndarray):
            # New frequency-first structure
            if self.time_summed:
                # Data shape: (num_freq_channels, num_antennas, num_antennas)
                # Decompose each frequency channel
                num_freq = self.n2_data.shape[0]
                eigenvals = np.zeros((num_freq, self.num_antennas), dtype=complex)
                eigenvecs = np.zeros((num_freq, self.num_antennas, self.num_antennas), dtype=complex)
                
                bar = _maybe_tqdm(num_freq, desc='Eigendecomposition by frequency')
                
                for f in range(num_freq):
                    evals, evecs = np.linalg.eigh(self.n2_data[f])
                    
                    # Sort by eigenvalue magnitude (descending)
                    idx = np.argsort(np.abs(evals))[::-1]
                    eigenvals[f] = evals[idx]
                    eigenvecs[f] = evecs[:, idx]
                    
                    if bar is not None:
                        bar.update(1)
                    else:
                        _console_progress(f + 1, num_freq, prefix='Eigendecomposition')
                
                if bar is not None:
                    bar.close()
                
                self.eigenvalues = eigenvals
                self.eigenvectors = eigenvecs
                self.eigendecomposition_metadata = {
                    'time_averaged': None,  # Already time-summed
                    'shape': eigenvals.shape,
                    'decomposition_type': 'frequency_resolved_time_summed'
                }
                
            else:
                # Data shape: (num_time_samples, num_freq_channels, num_antennas, num_antennas)
                if time_average:
                    # Average over time first, then decompose each frequency
                    n2_avg = np.mean(self.n2_data, axis=0)  # Shape: (num_freq_channels, num_antennas, num_antennas)
                    num_freq = n2_avg.shape[0]
                    eigenvals = np.zeros((num_freq, self.num_antennas), dtype=complex)
                    eigenvecs = np.zeros((num_freq, self.num_antennas, self.num_antennas), dtype=complex)
                    
                    bar = _maybe_tqdm(num_freq, desc='Eigendecomposition by frequency')
                    
                    for f in range(num_freq):
                        evals, evecs = np.linalg.eigh(n2_avg[f])
                        
                        # Sort by eigenvalue magnitude (descending)
                        idx = np.argsort(np.abs(evals))[::-1]
                        eigenvals[f] = evals[idx]
                        eigenvecs[f] = evecs[:, idx]
                        
                        if bar is not None:
                            bar.update(1)
                        else:
                            _console_progress(f + 1, num_freq, prefix='Eigendecomposition')
                    
                    if bar is not None:
                        bar.close()
                    
                    self.eigenvalues = eigenvals
                    self.eigenvectors = eigenvecs
                    self.eigendecomposition_metadata = {
                        'time_averaged': True,
                        'shape': eigenvals.shape,
                        'decomposition_type': 'frequency_resolved_time_averaged'
                    }
                    
                else:
                    # Decompose for each time and frequency
                    num_times, num_freq = self.n2_data.shape[:2]
                    eigenvals = np.zeros((num_times, num_freq, self.num_antennas), dtype=complex)
                    eigenvecs = np.zeros((num_times, num_freq, self.num_antennas, self.num_antennas), dtype=complex)
                    
                    total_iterations = num_times * num_freq
                    bar = _maybe_tqdm(total_iterations, desc='Eigendecomposition')
                    
                    iteration = 0
                    for t in range(num_times):
                        for f in range(num_freq):
                            evals, evecs = np.linalg.eigh(self.n2_data[t, f])
                            
                            # Sort by eigenvalue magnitude (descending)
                            idx = np.argsort(np.abs(evals))[::-1]
                            eigenvals[t, f] = evals[idx]
                            eigenvecs[t, f] = evecs[:, idx]
                            
                            iteration += 1
                            if bar is not None:
                                bar.update(1)
                            else:
                                _console_progress(iteration, total_iterations, prefix='Eigendecomposition')
                    
                    if bar is not None:
                        bar.close()
                    
                    self.eigenvalues = eigenvals
                    self.eigenvectors = eigenvecs
                    self.eigendecomposition_metadata = {
                        'time_averaged': False,
                        'shape': eigenvals.shape,
                        'decomposition_type': 'frequency_and_time_resolved'
                    }
        
        else:
            # Multiple frequency matrices case
            if freq_channels is None:
                freq_channels = self.freq_channels
            
            eigenvals_dict = {}
            eigenvecs_dict = {}
            
            bar = _maybe_tqdm(len(freq_channels), desc='Eigendecomposition by frequency')
            
            for i, freq_ch in enumerate(freq_channels):
                if freq_ch not in self.n2_data:
                    print(f"Warning: Frequency channel {freq_ch} not found in data")
                    continue
                
                n2_matrix = self.n2_data[freq_ch]
                
                if time_average:
                    # Average over time, then decompose
                    n2_avg = np.mean(n2_matrix, axis=0)
                    eigenvals, eigenvecs = np.linalg.eigh(n2_avg)
                    
                    # Sort by eigenvalue magnitude (descending)
                    idx = np.argsort(np.abs(eigenvals))[::-1]
                    eigenvals_dict[freq_ch] = eigenvals[idx]
                    eigenvecs_dict[freq_ch] = eigenvecs[:, idx]
                    
                else:
                    # Decompose for each time sample
                    num_times = n2_matrix.shape[0]
                    eigenvals = np.zeros((num_times, self.num_antennas), dtype=complex)
                    eigenvecs = np.zeros((num_times, self.num_antennas, self.num_antennas), dtype=complex)
                    
                    for t in range(num_times):
                        evals, evecs = np.linalg.eigh(n2_matrix[t])
                        
                        # Sort by eigenvalue magnitude (descending)
                        idx = np.argsort(np.abs(evals))[::-1]
                        eigenvals[t] = evals[idx]
                        eigenvecs[t] = evecs[:, idx]
                    
                    eigenvals_dict[freq_ch] = eigenvals
                    eigenvecs_dict[freq_ch] = eigenvecs
                
                if bar is not None:
                    bar.update(1)
                else:
                    _console_progress(i + 1, len(freq_channels), prefix='Eigendecomposition')
            
            if bar is not None:
                bar.close()
            
            self.eigenvalues = eigenvals_dict
            self.eigenvectors = eigenvecs_dict
            self.eigendecomposition_metadata = {
                'time_averaged': time_average,
                'frequencies_processed': freq_channels,
                'decomposition_type': f'multi_frequency_{"time_averaged" if time_average else "time_resolved"}'
            }
        
        self.eigendecomposition_done = True
        print("Eigenvalue decomposition completed!")
        
        return {
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'metadata': self.eigendecomposition_metadata
        }
    
    def plot_eigenvalues(self, num_eigenvalues=None, figsize=(12, 8), log_scale=True,
                        freq_channel=None, time_range=None, save_path=None):
        """
        Plot eigenvalues from the decomposition.
        
        Parameters
        ----------
        num_eigenvalues : int or None
            Number of largest eigenvalues to plot. If None, plots all.
        figsize : tuple
            Figure size (width, height)
        log_scale : bool
            Whether to use log scale for eigenvalue magnitude
        freq_channel : int or None
            For multi-frequency data, which frequency channel to plot.
            If None, plots all or prompts user.
        time_range : tuple or None
            (start_time, stop_time) for time-resolved plots
        save_path : str or None
            Path to save the plot
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if not self.eigendecomposition_done:
            raise RuntimeError("Must call compute_eigendecomposition() first!")
        
        if num_eigenvalues is None:
            num_eigenvalues = self.num_antennas
        else:
            num_eigenvalues = min(num_eigenvalues, self.num_antennas)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        if self.sum_frequencies:
            # Single matrix case
            eigenvals = self.eigenvalues
            
            if self.eigendecomposition_metadata['time_averaged']:
                # Time-averaged eigenvalues
                evals_magnitude = np.abs(eigenvals[:num_eigenvalues])
                evals_phase = np.angle(eigenvals[:num_eigenvalues])
                
                # Plot magnitude
                axes[0].bar(range(num_eigenvalues), evals_magnitude)
                axes[0].set_xlabel('Eigenvalue Index')
                axes[0].set_ylabel('Eigenvalue Magnitude')
                axes[0].set_title(f'Eigenvalue Magnitudes (Top {num_eigenvalues})')
                if log_scale:
                    axes[0].set_yscale('log')
                
                # Plot phase
                axes[1].bar(range(num_eigenvalues), evals_phase)
                axes[1].set_xlabel('Eigenvalue Index')
                axes[1].set_ylabel('Eigenvalue Phase (radians)')
                axes[1].set_title(f'Eigenvalue Phases (Top {num_eigenvalues})')
                
            else:
                # Time-resolved eigenvalues
                evals_magnitude = np.abs(eigenvals[:, :num_eigenvalues])
                
                if time_range is not None and self.times is not None:
                    start_idx = np.searchsorted(self.times, time_range[0])
                    stop_idx = np.searchsorted(self.times, time_range[1])
                    time_slice = slice(start_idx, stop_idx)
                    time_axis = self.times[time_slice]
                    evals_magnitude = evals_magnitude[time_slice, :]
                else:
                    time_axis = np.arange(evals_magnitude.shape[0])
                
                # Plot magnitude vs time for top eigenvalues
                for i in range(num_eigenvalues):
                    axes[0].plot(time_axis, evals_magnitude[:, i], label=f'λ_{i}')
                
                axes[0].set_xlabel('Time' + (' (s)' if self.times is not None else ' (samples)'))
                axes[0].set_ylabel('Eigenvalue Magnitude')
                axes[0].set_title(f'Eigenvalue Magnitudes vs Time (Top {num_eigenvalues})')
                axes[0].legend()
                if log_scale:
                    axes[0].set_yscale('log')
                
                # Plot eigenvalue ratios
                if num_eigenvalues > 1:
                    ratio = evals_magnitude[:, 0] / evals_magnitude[:, 1]
                    axes[1].plot(time_axis, ratio)
                    axes[1].set_xlabel('Time' + (' (s)' if self.times is not None else ' (samples)'))
                    axes[1].set_ylabel('λ₀/λ₁ Ratio')
                    axes[1].set_title('Ratio of Largest to Second Largest Eigenvalue')
                else:
                    axes[1].text(0.5, 0.5, 'Need at least 2 eigenvalues for ratio plot',
                               ha='center', va='center', transform=axes[1].transAxes)
        
        else:
            # Multi-frequency case
            if freq_channel is None:
                if len(self.freq_channels) == 1:
                    freq_channel = self.freq_channels[0]
                else:
                    print(f"Multiple frequencies available: {self.freq_channels}")
                    print("Please specify freq_channel parameter")
                    return fig
            
            if freq_channel not in self.eigenvalues:
                raise ValueError(f"Frequency channel {freq_channel} not found in eigenvalue data")
            
            eigenvals = self.eigenvalues[freq_channel]
            frequency_mhz = self.frequencies[self.freq_channels.index(freq_channel)]
            
            if self.eigendecomposition_metadata['time_averaged']:
                # Time-averaged eigenvalues
                evals_magnitude = np.abs(eigenvals[:num_eigenvalues])
                evals_phase = np.angle(eigenvals[:num_eigenvalues])
                
                # Plot magnitude
                axes[0].bar(range(num_eigenvalues), evals_magnitude)
                axes[0].set_xlabel('Eigenvalue Index')
                axes[0].set_ylabel('Eigenvalue Magnitude')
                axes[0].set_title(f'Eigenvalue Magnitudes at {frequency_mhz:.1f} MHz (Top {num_eigenvalues})')
                if log_scale:
                    axes[0].set_yscale('log')
                
                # Plot phase
                axes[1].bar(range(num_eigenvalues), evals_phase)
                axes[1].set_xlabel('Eigenvalue Index')
                axes[1].set_ylabel('Eigenvalue Phase (radians)')
                axes[1].set_title(f'Eigenvalue Phases at {frequency_mhz:.1f} MHz (Top {num_eigenvalues})')
                
            else:
                # Time-resolved eigenvalues
                evals_magnitude = np.abs(eigenvals[:, :num_eigenvalues])
                
                if time_range is not None and self.times is not None:
                    start_idx = np.searchsorted(self.times, time_range[0])
                    stop_idx = np.searchsorted(self.times, time_range[1])
                    time_slice = slice(start_idx, stop_idx)
                    time_axis = self.times[time_slice]
                    evals_magnitude = evals_magnitude[time_slice, :]
                else:
                    time_axis = np.arange(evals_magnitude.shape[0])
                
                # Plot magnitude vs time for top eigenvalues
                for i in range(num_eigenvalues):
                    axes[0].plot(time_axis, evals_magnitude[:, i], label=f'λ_{i}')
                
                axes[0].set_xlabel('Time' + (' (s)' if self.times is not None else ' (samples)'))
                axes[0].set_ylabel('Eigenvalue Magnitude')
                axes[0].set_title(f'Eigenvalue Magnitudes vs Time at {frequency_mhz:.1f} MHz (Top {num_eigenvalues})')
                axes[0].legend()
                if log_scale:
                    axes[0].set_yscale('log')
                
                # Plot eigenvalue ratios
                if num_eigenvalues > 1:
                    ratio = evals_magnitude[:, 0] / evals_magnitude[:, 1]
                    axes[1].plot(time_axis, ratio)
                    axes[1].set_xlabel('Time' + (' (s)' if self.times is not None else ' (samples)'))
                    axes[1].set_ylabel('λ₀/λ₁ Ratio')
                    axes[1].set_title(f'Ratio of Largest to Second Largest Eigenvalue at {frequency_mhz:.1f} MHz')
                else:
                    axes[1].text(0.5, 0.5, 'Need at least 2 eigenvalues for ratio plot',
                               ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def get_dominant_mode(self, freq_channel_index=None, time_index=None):
        """
        Get the dominant eigenmode (largest eigenvalue and corresponding eigenvector).
        
        Parameters
        ----------
        freq_channel_index : int or None
            Which frequency channel index to use (0 to num_frequencies-1).
            If None and multiple frequencies, returns all frequencies.
        time_index : int or None
            For time-resolved data, which time index to use.
            If None, uses all times or time-averaged data.
            
        Returns
        -------
        dict
            Dictionary with dominant eigenvalue and eigenvector
        """
        if not self.eigendecomposition_done:
            raise RuntimeError("Must call compute_eigendecomposition() first!")
        
        decomp_type = self.eigendecomposition_metadata['decomposition_type']
        
        if decomp_type == 'frequency_resolved_time_summed':
            # Shape: (num_freq, num_antennas)
            if freq_channel_index is None:
                # Return all frequencies
                dominant_eigenval = self.eigenvalues[:, 0]  # Shape: (num_freq,)
                dominant_eigenvec = self.eigenvectors[:, :, 0]  # Shape: (num_freq, num_antennas)
            else:
                if freq_channel_index >= self.eigenvalues.shape[0]:
                    raise ValueError(f"freq_channel_index {freq_channel_index} >= {self.eigenvalues.shape[0]}")
                dominant_eigenval = self.eigenvalues[freq_channel_index, 0]
                dominant_eigenvec = self.eigenvectors[freq_channel_index, :, 0]
        
        elif decomp_type == 'frequency_resolved_time_averaged':
            # Shape: (num_freq, num_antennas) 
            if freq_channel_index is None:
                # Return all frequencies
                dominant_eigenval = self.eigenvalues[:, 0]  # Shape: (num_freq,)
                dominant_eigenvec = self.eigenvectors[:, :, 0]  # Shape: (num_freq, num_antennas)
            else:
                if freq_channel_index >= self.eigenvalues.shape[0]:
                    raise ValueError(f"freq_channel_index {freq_channel_index} >= {self.eigenvalues.shape[0]}")
                dominant_eigenval = self.eigenvalues[freq_channel_index, 0]
                dominant_eigenvec = self.eigenvectors[freq_channel_index, :, 0]
        
        elif decomp_type == 'frequency_and_time_resolved':
            # Shape: (num_times, num_freq, num_antennas)
            if time_index is None and freq_channel_index is None:
                # Return all times and frequencies
                dominant_eigenval = self.eigenvalues[:, :, 0]  # Shape: (num_times, num_freq)
                dominant_eigenvec = self.eigenvectors[:, :, :, 0]  # Shape: (num_times, num_freq, num_antennas)
            elif time_index is None:
                # All times, specific frequency
                if freq_channel_index >= self.eigenvalues.shape[1]:
                    raise ValueError(f"freq_channel_index {freq_channel_index} >= {self.eigenvalues.shape[1]}")
                dominant_eigenval = self.eigenvalues[:, freq_channel_index, 0]  # Shape: (num_times,)
                dominant_eigenvec = self.eigenvectors[:, freq_channel_index, :, 0]  # Shape: (num_times, num_antennas)
            elif freq_channel_index is None:
                # Specific time, all frequencies
                if time_index >= self.eigenvalues.shape[0]:
                    raise ValueError(f"time_index {time_index} >= {self.eigenvalues.shape[0]}")
                dominant_eigenval = self.eigenvalues[time_index, :, 0]  # Shape: (num_freq,)
                dominant_eigenvec = self.eigenvectors[time_index, :, :, 0]  # Shape: (num_freq, num_antennas)
            else:
                # Specific time and frequency
                if time_index >= self.eigenvalues.shape[0]:
                    raise ValueError(f"time_index {time_index} >= {self.eigenvalues.shape[0]}")
                if freq_channel_index >= self.eigenvalues.shape[1]:
                    raise ValueError(f"freq_channel_index {freq_channel_index} >= {self.eigenvalues.shape[1]}")
                dominant_eigenval = self.eigenvalues[time_index, freq_channel_index, 0]
                dominant_eigenvec = self.eigenvectors[time_index, freq_channel_index, :, 0]
        
        else:
            raise RuntimeError(f"Unknown decomposition type: {decomp_type}")
        
        return {
            'eigenvalue': dominant_eigenval,
            'eigenvector': dominant_eigenvec,
            'antennas': self.antennas,
            'freq_channel_index': freq_channel_index,
            'time_index': time_index,
            'decomposition_type': decomp_type
        }
    
    def summary(self):
        """Print a summary of the N2Matrix object."""
        print("=" * 50)
        print("N2 Matrix Summary")
        print("=" * 50)
        print(f"Antennas: {self.antennas}")
        print(f"Number of antennas: {self.num_antennas}")
        print(f"Frequencies: {self.num_frequencies} channels")
        print(f"Frequency range: {min(self.frequencies):.1f} - {max(self.frequencies):.1f} MHz")
        print(f"Polarization: {self.polarization}")
        print(f"Sum frequencies: {self.sum_frequencies}")
        print(f"Data shape: {self.shape}")
        print(f"Number of time samples: {self.num_time_samples}")
        
        if self.times is not None:
            print(f"Time range: {self.times[0]:.6f} - {self.times[-1]:.6f} s")
            print(f"Time resolution: {np.mean(np.diff(self.times)):.6f} s")
        
        if self.eigendecomposition_done:
            print("\nEigenvalue Decomposition:")
            print(f"  Type: {self.eigendecomposition_metadata['decomposition_type']}")
            print(f"  Time averaged: {self.eigendecomposition_metadata['time_averaged']}")
            if isinstance(self.eigenvalues, dict):
                print(f"  Frequencies processed: {len(self.eigenvalues)}")
            else:
                print(f"  Eigenvalue shape: {self.eigenvalues.shape}")
        else:
            print("\nEigenvalue decomposition: Not computed")
        
        print("=" * 50)


# ============================================================================
# BASE CLASS FOR ANTENNA FLAGGING AND DATA MANAGEMENT
# ============================================================================

class BaseAntennaProcessor(ABC):
    """Base class for antenna data processing with common flagging functionality."""
    
    def __init__(self, directory, ant_flagged=None):
        self.directory = directory
        self.ant_flagged = ant_flagged if ant_flagged is not None else []
        self._data_loaded = False
        
        # Common data storage
        self.all_data = {}
        self.all_meta = {}
        self.summed_data = None
        self.sum_metadata = None
    
    # Antenna flagging methods
    def update_flags(self, new_flagged_antennas):
        """Update the list of flagged antennas"""
        self.ant_flagged = list(new_flagged_antennas)
        print(f"Updated flagged antennas: {sorted(self.ant_flagged)}")
        self._clear_cached_sums()
    
    def add_flagged_antennas(self, antennas_to_flag):
        """Add antennas to the flagged list"""
        if isinstance(antennas_to_flag, int):
            antennas_to_flag = [antennas_to_flag]
            
        for ant in antennas_to_flag:
            if ant not in self.ant_flagged:
                self.ant_flagged.append(ant)
        
        print(f"Added antennas {antennas_to_flag} to flagged list")
        print(f"Current flagged antennas: {sorted(self.ant_flagged)}")
        self._clear_cached_sums()
    
    def remove_flagged_antennas(self, antennas_to_unflag):
        """Remove antennas from the flagged list"""
        if isinstance(antennas_to_unflag, int):
            antennas_to_unflag = [antennas_to_unflag]
            
        for ant in antennas_to_unflag:
            if ant in self.ant_flagged:
                self.ant_flagged.remove(ant)
        
        print(f"Removed antennas {antennas_to_unflag} from flagged list")
        print(f"Current flagged antennas: {sorted(self.ant_flagged)}")
        self._clear_cached_sums()
    
    def get_unflagged_antennas(self):
        """Get list of unflagged antenna numbers"""
        all_ants = self.get_antenna_list()
        return sorted([ant for ant in all_ants if ant not in self.ant_flagged])
    
    def get_flagged_antennas(self):
        """Get list of flagged antenna numbers"""
        return sorted(self.ant_flagged)
    
    def _clear_cached_sums(self):
        """Clear cached summed data when flags change"""
        self.summed_data = None
        self.sum_metadata = None
    
    # Abstract methods
    @abstractmethod
    def get_antenna_list(self):
        """Get list of available antenna numbers"""
        pass
    
    @abstractmethod
    def load_all_data(self):
        """Load all available data"""
        pass
    
    @abstractmethod
    def print_summary(self):
        """Print a summary of loaded data and current flags"""
        pass

# ============================================================================
# TIME SERIES PLOTTER CLASS
# ============================================================================

class TimeSeries(BaseAntennaProcessor):
    """A class for analyzing time series data."""
    
    def __init__(self, files_dir, ant_flagged=None):
        super().__init__(files_dir, ant_flagged)
        self.files_dir = files_dir
        
        # Specific to DedispAnalyzer
        self.metas = []
        self.data = {}
        self.num_antennas = 0
    
    def _clear_cached_sums(self):
        """Clear cached summed data when flags change"""
        super()._clear_cached_sums()
        self.metas = []
        self.data = {}
        self.num_antennas = 0
    
    def read_dedisp(self, filename):
        """Read a single .dedisp file and extract metadata and data."""
        with open(filename, "rb") as f:
            header = f.read(4096).decode("utf-8", "replace")
            meta = {}
            for line in header.splitlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                if parts[0] == "ANT":
                    meta['ant'] = int(parts[1])
                if parts[0] == "DM":
                    meta['dm'] = float(parts[1])
                if parts[0] == "NOUT":
                    meta["nout"] = int(parts[1])
                if parts[0] == "TSAMP":
                    meta["tsamp"] = float(parts[1])
                if parts[0] == "NBLOCKS":
                    meta["nblocks"] = int(parts[1])
                if parts[0] == "NTIME":
                    meta["ntime"] = int(parts[1])
                if parts[0] == "NPOL":
                    meta["npol"] = int(parts[1])
                if parts[0] == "START_SAMPLE":
                    meta["start_sample"] = int(parts[1])
                if parts[0] == "START_TIME":
                    meta["start_time"] = float(parts[1])
        
        # Set defaults
        if "start_sample" not in meta:
            meta["start_sample"] = 0
        if "start_time" not in meta:
            meta["start_time"] = 0.0
            
        nout = meta["nout"]
        data = np.fromfile(filename, dtype=np.float32, offset=4096, count=nout)
        return meta, data
    
    def load_all_data(self):
        """Load ALL .dedisp files in the directory"""
        files = glob(f"{self.files_dir}/*.dedisp")
        files.sort()
        
        self.all_data = {}
        self.all_meta = {}
        
        print(f"Loading {len(files)} antenna files...")
        for fn in files:
            try:
                ant = int(fn.split('_')[-1][:2])
                meta, arr = self.read_dedisp(fn)
                self.all_meta[ant] = meta
                self.all_data[ant] = arr
            except Exception as e:
                print(f"Error loading {fn}: {e}")
                continue
        
        self._data_loaded = True
        loaded_ants = sorted(self.all_data.keys())
        if loaded_ants:
            print(f"Successfully loaded {len(loaded_ants)} antennas: {loaded_ants[0]} to {loaded_ants[-1]}")
        else:
            print("No antennas loaded successfully")
        
        # Validate consistency
        if not self.all_meta:
            raise RuntimeError("No valid data files found!")
            
        nout_set = set(meta['nout'] for meta in self.all_meta.values())
        tsamp_set = set(meta['tsamp'] for meta in self.all_meta.values())
        dm_set = set(np.round(meta['dm'], 6) for meta in self.all_meta.values())
        
        if len(nout_set) != 1 or len(tsamp_set) != 1 or len(dm_set) != 1:
            raise RuntimeError("Mismatch in NOUT, TSAMP, or DM between files!")
        
        return self.all_data, self.all_meta
    
    def get_antenna_list(self):
        """Get list of available antenna numbers"""
        if self._data_loaded:
            return sorted(self.all_data.keys())
            
        files = glob(f"{self.files_dir}/*.dedisp")
        ants = []
        for fn in files:
            try:
                ant = int(fn.split('_')[-1][:2])
                ants.append(ant)
            except ValueError:
                continue
        return sorted(ants)
    
    def time_to_indices(self, start_time=None, stop_time=None, tsamp=None, start_time_header=0.0):
        """Convert time values in seconds to array indices.

        Args:
            start_time (float, optional): Start time in seconds (None for beginning)
            stop_time (float, optional): Stop time in seconds (None for end)
            tsamp (float, optional): Time sampling in seconds (will use metadata if None)
            start_time_header (float): Header start time offset

        Returns:
            tuple: (start_idx, stop_idx)
        """
        if tsamp is None:
            if self._data_loaded and self.all_meta:
                tsamp = list(self.all_meta.values())[0]['tsamp']
            elif self.metas:
                tsamp = self.metas[0]['tsamp']
            else:
                raise RuntimeError("No metadata available. Load data first or provide tsamp.")
        
        # FIXED: Convert times to indices relative to the header start time
        # This ensures consistency between slicing and display
        start_idx = 0 if start_time is None else int((start_time - start_time_header) / tsamp)
        stop_idx = -1 if stop_time is None else int((stop_time - start_time_header) / tsamp)

        return start_idx, stop_idx
    
    def integrate_data(self, data, times, int_time_samples, start_idx=0, stop_idx=-1):
        """Integrate data over time samples"""
        return integrate_data_arrays(data, times, int_time_samples, start_idx, stop_idx)
    
    def integrate_data_time_slice(self, data, times, int_time_samples, start_time=None, stop_time=None):
        """Integrate data over time samples using time-based slicing.

        Args:
            data (np.array): Input data
            times (np.array): Time array
            int_time_samples (int): Number of samples to integrate
            start_time (float, optional): Start time in seconds (None for beginning)
            stop_time (float, optional): Stop time in seconds (None for end)

        Returns:
            tuple: (integrated_data, integrated_times)
        """
        tsamp = times[1] - times[0] if len(times) > 1 else times[0]
        # FIXED: Use the start time from the times array for consistent conversion
        start_time_header = times[0] if len(times) > 0 else 0.0
        start_idx, stop_idx = self.time_to_indices(
            start_time, stop_time, tsamp, start_time_header)

        return self.integrate_data(data, times, int_time_samples, start_idx, stop_idx)
    
    def calculate_snr(self, data):
        """Calculate SNR using MAD"""
        return calculate_snr(data)
    
    def compute_sum(self):
        """Compute the sum of unflagged antennas"""
        if not self._data_loaded:
            raise RuntimeError("Must call load_all_data() first!")
        
        # Reset unflagged data structures
        self.metas = []
        self.data = {}
        datas = []
        self.num_antennas = 0
        
        available_ants = sorted(self.all_data.keys())
        print(f"Available antennas: {available_ants}")
        print(f"Flagged antennas: {sorted(self.ant_flagged)}")
        
        for ant in available_ants:
            if ant in self.ant_flagged:
                print(f"Excluding flagged antenna {ant}")
                continue
                
            meta = self.all_meta[ant]
            arr = self.all_data[ant]
            
            self.metas.append(meta)
            datas.append(arr)
            self.data[ant] = arr
            self.num_antennas += 1
        
        if not datas:
            raise RuntimeError("No unflagged antennas available for summation!")
        
        print(f"Summing {self.num_antennas} unflagged antennas")
        stacked = np.vstack(datas)
        self.summed_data = stacked.sum(axis=0)
        
        return self.summed_data, self.metas[0], self.num_antennas
    
    def plot_snr_vs_time(self, int_time_samples=1, start_idx=0, stop_idx=-1,
                         start_time=None, stop_time=None):
        """Plot SNR vs time for summed antenna data"""
        if self.summed_data is None:
            if not self._data_loaded:
                print("Loading all data first...")
                self.load_all_data()
            self.compute_sum()

        tsamp = self.metas[0]['tsamp']
        # IMPROVED: Calculate correct time axis from header information
        start_time_header = self.metas[0].get('start_time', 0.0)
        start_sample_header = self.metas[0].get('start_sample', 0)

        # The time axis should start from start_time_header and increment by tsamp
        # This represents the actual time in the original observation
        times = start_time_header + np.arange(len(self.summed_data)) * tsamp

        print(f"Time sampling: {tsamp} s")
        print(
            f"Header start_time: {start_time_header} s, start_sample: {start_sample_header}")
        print(
            f"Time range: {times[0]:.6f} to {times[-1]:.6f} s ({len(times)} samples)")

        # Create time array starting from the actual start time
        times = start_time_header + np.arange(len(self.summed_data)) * tsamp

        # Use time-based slicing if times are provided
        if start_time is not None or stop_time is not None:
            data_integrated, times_integrated = self.integrate_data_time_slice(
                self.summed_data, times, int_time_samples, start_time, stop_time
            )
            # Convert back to indices for display
            start_idx_actual = 0 if start_time is None else int(
                (start_time - start_time_header) / tsamp)
            stop_idx_actual = len(
                self.summed_data)-1 if stop_time is None else int((stop_time - start_time_header) / tsamp)
            time_info = f"Time: {start_time if start_time is not None else start_time_header:.5f}s to " + \
                f"{stop_time if stop_time is not None else times[-1]:.5f}s " + \
                f"(indices: {start_idx_actual} to {stop_idx_actual})"
        else:
            data_integrated, times_integrated = self.integrate_data(
                self.summed_data, times, int_time_samples, start_idx, stop_idx
            )
            # Convert indices to times for display
            start_time_actual = start_time_header + start_idx * tsamp
            stop_time_actual = times[-1] if stop_idx == - \
                1 else start_time_header + stop_idx * tsamp
            stop_idx_actual = len(self.summed_data) - \
                1 if stop_idx == -1 else stop_idx
            time_info = f"Indices: {start_idx} to {stop_idx_actual} " + \
                f"(time: {start_time_actual:.5f}s to {stop_time_actual:.5f}s)"

        snr, median_val, std, p = self.calculate_snr(data_integrated)
        print(p)

        plt.figure(figsize=(12, 6))
        plt.step(times_integrated, snr, where='mid', color='k')
        plt.xlabel("Time (s)")
        plt.ylabel("SNR")
        plt.title(f"Summed over {self.num_antennas} antennas, DM={self.metas[0]['dm']:.2f}\n"
                  f"Time samples integrated = {int_time_samples}, {time_info}")

        # Add pulse visualizations
        for start_idx, stop_idx in p:
            start_time = times_integrated[start_idx]
            stop_time = times_integrated[stop_idx]
            plt.axvline(start_time, color='black', linestyle='--', alpha=0.5)
            plt.axvline(stop_time, color='black', linestyle='--', alpha=0.5)
            plt.axvspan(start_time, stop_time, color='grey', alpha=0.1)

        plt.show()
        return snr, times_integrated
    
    def plot_snr_vs_time_ant(self, ant, int_time_samples=1, start_idx=0, stop_idx=-1,
                             start_time=None, stop_time=None):
        """Plot SNR vs time for a specific antenna"""
        # Use cached data if available, otherwise load from file
        if self._data_loaded and ant in self.all_data:
            meta = self.all_meta[ant]
            arr = self.all_data[ant]
            print(f"Using cached data for antenna {ant}")
        else:
            filename = f"{self.files_dir}/ant_{ant:02d}.dedisp"
            print(f"Loading: {filename}")
            meta, arr = self.read_dedisp(filename)

        tsamp = meta['tsamp']
        # IMPROVED: Calculate correct time axis from header information
        start_time_header = meta.get('start_time', 0.0)
        start_sample_header = meta.get('start_sample', 0)

        # The time axis should start from start_time_header and increment by tsamp
        times = start_time_header + np.arange(len(arr)) * tsamp

        print(f"Antenna {meta['ant']}: Time sampling: {tsamp} s")
        print(
            f"Header start_time: {start_time_header} s, start_sample: {start_sample_header}")
        print(
            f"Time range: {times[0]:.6f} to {times[-1]:.6f} s ({len(times)} samples)")

        # Use time-based slicing if times are provided
        if start_time is not None or stop_time is not None:
            data_integrated, times_integrated = self.integrate_data_time_slice(
                arr, times, int_time_samples, start_time, stop_time
            )
            # Convert back to indices for display
            start_idx_actual = 0 if start_time is None else int(
                (start_time - start_time_header) / tsamp)
            stop_idx_actual = len(
                arr)-1 if stop_time is None else int((stop_time - start_time_header) / tsamp)
            time_info = f"Time: {start_time if start_time is not None else start_time_header:.1f}s to " + \
                f"{stop_time if stop_time is not None else times[-1]:.1f}s " + \
                f"(indices: {start_idx_actual} to {stop_idx_actual})"
        else:
            data_integrated, times_integrated = self.integrate_data(
                arr, times, int_time_samples, start_idx, stop_idx
            )
            # Convert indices to times for display
            start_time_actual = start_time_header + start_idx * tsamp
            stop_time_actual = times[-1] if stop_idx == - \
                1 else start_time_header + stop_idx * tsamp
            stop_idx_actual = len(arr)-1 if stop_idx == -1 else stop_idx
            time_info = f"Indices: {start_idx} to {stop_idx_actual} " + \
                f"(time: {start_time_actual:.5f}s to {stop_time_actual:.5f}s)"

        snr, _, _, p = self.calculate_snr(data_integrated)

        plt.figure(figsize=(12, 6))
        plt.step(times_integrated, snr, where='mid', color='k')
        plt.xlabel("Time (s)")
        plt.ylabel("SNR")
        plt.title(f"Antenna {meta['ant']}, DM={meta['dm']:.2f}\n"
                  f"Time samples integrated = {int_time_samples}, {time_info}")

        # Add pulse visualizations
        for start_idx, stop_idx in p:
            start_time = times_integrated[start_idx]
            stop_time = times_integrated[stop_idx]
            plt.axvline(start_time, color='black', linestyle='--', alpha=0.5)
            plt.axvline(stop_time, color='black', linestyle='--', alpha=0.5)
            plt.axvspan(start_time, stop_time, color='grey', alpha=0.1)

        plt.show()

        return snr, times_integrated
    
    def plot_all_ants(self, ants=None, nrow=4, ncol=4, int_time_samples=1, start_idx=0, stop_idx=-1,
                      start_time=None, stop_time=None):
        """
        Plot SNR vs time for multiple antennas in a grid layout.

        Args:
            ants (list): List of antenna numbers to plot (if None, plots all available)
            nrow (int): Number of rows in subplot grid
            ncol (int): Number of columns in subplot grid
            int_time_samples (int): Number of time samples to integrate
            start_idx (int): Start index for time slice (ignored if start_time given)
            stop_idx (int): Stop index for time slice (ignored if stop_time given)
            start_time (float, optional): Start time in seconds
            stop_time (float, optional): Stop time in seconds
        """
        if ants is None:
            ants = self.get_antenna_list()

        num_ants = len(ants)
        panels_per_page = nrow * ncol
        num_pages = (num_ants + panels_per_page - 1) // panels_per_page

        # Determine time info for title
        if start_time is not None or stop_time is not None:
            # Get tsamp and start_time_header for conversion
            if self._data_loaded and self.all_meta:
                tsamp = list(self.all_meta.values())[0]['tsamp']
                start_time_header = list(self.all_meta.values())[0].get('start_time', 0.0)
            else:
                tsamp = 1.0  # fallback
                start_time_header = 0.0
            # FIXED: Convert times relative to header start time
            start_idx_actual = 0 if start_time is None else int(
                (start_time - start_time_header) / tsamp)
            stop_idx_actual = -1 if stop_time is None else int(
                (stop_time - start_time_header) / tsamp)
            time_info = f"Time: {start_time if start_time is not None else start_time_header:.1f}s to " + \
                f"{stop_time if stop_time is not None else 'end'}s " + \
                f"(indices: {start_idx_actual} to {stop_idx_actual if stop_idx_actual != -1 else 'end'})"
        else:
            # Get tsamp and start_time_header for conversion
            if self._data_loaded and self.all_meta:
                tsamp = list(self.all_meta.values())[0]['tsamp']
                start_time_header = list(self.all_meta.values())[0].get('start_time', 0.0)
            else:
                tsamp = 1.0  # fallback
                start_time_header = 0.0
            # FIXED: Convert indices to absolute times using header information
            start_time_actual = start_time_header + start_idx * tsamp
            stop_time_actual = 'end' if stop_idx == -1 else start_time_header + stop_idx * tsamp
            time_info = f"Indices: {start_idx} to {stop_idx if stop_idx != -1 else 'end'} " + \
                f"(time: {start_time_actual:.1f}s to {stop_time_actual if isinstance(stop_time_actual, str) else f'{stop_time_actual:.1f}s'})"

        for page in range(num_pages):
            fig, axes = plt.subplots(nrow, ncol, figsize=(
                ncol*4, nrow*2.5), squeeze=False)

            for p in range(panels_per_page):
                ant_idx = page * panels_per_page + p
                if ant_idx >= num_ants:
                    axes.flat[p].axis('off')
                    continue

                ax = axes.flat[p]
                ant = ants[ant_idx]

                try:
                    # Use cached data if available, otherwise load from file
                    if self._data_loaded and ant in self.all_data:
                        meta = self.all_meta[ant]
                        arr = self.all_data[ant]
                    else:
                        filename = f"{self.files_dir}/ant_{ant:02d}.dedisp"
                        meta, arr = self.read_dedisp(filename)
                except Exception as e:
                    print(f"Error reading antenna {ant}: {e}")
                    ax.axis('off')
                    continue

                tsamp = meta['tsamp']
                # FIXED: Calculate correct time axis from header information (like other plotting functions)
                start_time_header = meta.get('start_time', 0.0)
                times = start_time_header + np.arange(len(arr)) * tsamp

                # Use time-based slicing if times are provided
                if start_time is not None or stop_time is not None:
                    data_integrated, times_integrated = self.integrate_data_time_slice(
                        arr, times, int_time_samples, start_time, stop_time
                    )
                else:
                    data_integrated, times_integrated = self.integrate_data(
                        arr, times, int_time_samples, start_idx, stop_idx
                    )

                snr, _, _, _ = self.calculate_snr(data_integrated)

                # Color code flagged antennas
                color = 'red' if ant in self.ant_flagged else 'black'
                ax.step(times_integrated, snr, where='mid', color=color)

                title = f"Ant {meta['ant']:02d}"
                if ant in self.ant_flagged:
                    title += " (FLAGGED)"
                ax.set_title(title)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("SNR")

            plt.tight_layout()
            plt.suptitle(f"Antennas {ants[page*panels_per_page]} to "
                         f"{ants[min((page+1)*panels_per_page-1, num_ants-1)]} "
                         f"Time samples integrated: {int_time_samples}\n"
                         f"Stokes-I summed over all freq channels "
                         f"(Red=Flagged, Black=Unflagged), {time_info}", y=1.02)
            plt.subplots_adjust(top=0.88)
            plt.show()
    
    def print_summary(self):
        """Print a summary of loaded data and current flags"""
        all_ants = self.get_available_antennas()
        flagged_ants = self.get_flagged_antennas()
        unflagged_ants = self.get_unflagged_antennas()
        
        print("Data Summary:")
        print(f"  Directory: {self.directory}")
        print(f"  Total antennas available: {len(all_ants)}")
        print(f"  Available antennas: {all_ants}")
        print(f"  Flagged antennas: {flagged_ants} ({len(flagged_ants)} total)")
        print(f"  Unflagged antennas: {unflagged_ants} ({len(unflagged_ants)} total)")
        
        if self._data_loaded and self.all_meta:
            meta = list(self.all_meta.values())[0]
            print(f"  Time samples: {meta['time_samples']}")
            print(f"  Total channels: {meta['total_channels']}")
            print(f"  Number of subbands: {meta['num_subbands']}")
            print(f"  Total observation time: {meta['time_samples'] * meta['sample_time']:.2f} s")


# ============================================================================
# WATERFALL PLOTTER CLASS
# ============================================================================

cmap = 'magma'

class DynamicSpectra(BaseAntennaProcessor):
    """DSA110 spectra waterfall plotter with antenna flagging capabilities"""
    
    def __init__(self, directory, ant_flagged=None):
        super().__init__(directory, ant_flagged)
        
        # DSA110 frequency parameters
        self.subband_start_freqs = [
            1498.75, 1487.03125, 1475.3125, 1463.59375, 1451.875, 1440.15625,
            1428.4375, 1416.71875, 1405.0, 1393.28125, 1381.5625, 1369.84375,
            1358.125, 1346.40625, 1334.6875, 1322.96875
        ]
        self.channel_freq_step = -0.030517578125
        self.channels_per_subband = 384
        self.time_step = 8.192e-6
    
    def _find_antenna_files(self, antenna):
        """Find all subband files for given antenna"""
        patterns = [
            f"ant_{antenna:02d}_subband_*_stokes_i.spec",
            f"ant_{antenna}_subband_*_stokes_i.spec",
            f"*ant{antenna:02d}*subband*.spec",
            f"*ant{antenna}_*subband*.spec",
        ]
        
        files = []
        for pattern in patterns:
            found = glob(os.path.join(self.directory, pattern))
            files.extend(found)
            if found:
                break
        
        return sorted(list(set(files)))
    
    def _read_spectra_file(self, filename):
        """Read a single binary spectra file"""
        with open(filename, 'rb') as f:
            header_data = f.read(24)
            
            if len(header_data) == 24:
                header = struct.unpack('IIIIII', header_data)
                num_time_samples, num_channels, antenna_idx, subband_idx, start_time_us, sample_time_ns = header
                start_time_sec = start_time_us / 1000000.0
                sample_time_sec = sample_time_ns / 1000000000.0
            elif len(header_data) == 16:
                f.seek(0)
                header_data = f.read(16)
                header = struct.unpack('IIII', header_data)
                num_time_samples, num_channels, antenna_idx, subband_idx = header
                start_time_sec = 0.0
                sample_time_sec = self.time_step
            else:
                raise ValueError(f"Invalid header size in {filename}: {len(header_data)} bytes")
            
            data_size = num_time_samples * num_channels * 4
            data_bytes = f.read(data_size)
            
            if len(data_bytes) != data_size:
                raise ValueError(f"Data size mismatch in {filename}")
            
            flat_data = struct.unpack(f'{num_time_samples * num_channels}f', data_bytes)
            data = np.array(flat_data).reshape(num_time_samples, num_channels)
            
            return {
                'data': data,
                'antenna': antenna_idx,
                'subband': subband_idx,
                'time_samples': num_time_samples,
                'channels': num_channels,
                'start_time': start_time_sec,
                'sample_time': sample_time_sec
            }
    
    def _calculate_frequencies(self, num_subbands):
        """Calculate frequency array for all subbands"""
        total_channels = num_subbands * self.channels_per_subband
        frequencies = np.zeros(total_channels)
        
        for sb in range(num_subbands):
            start_freq = self.subband_start_freqs[sb]
            start_idx = sb * self.channels_per_subband
            
            for ch in range(self.channels_per_subband):
                frequencies[start_idx + ch] = start_freq + (ch * self.channel_freq_step)
        
        return frequencies
    
    def _load_antenna_data(self, antenna):
        """Load and process data for a single antenna"""
        files = self._find_antenna_files(antenna)
        if not files:
            return None, None
        
        subband_data = {}
        for filename in files:
            try:
                data_info = self._read_spectra_file(filename)
                subband_data[data_info['subband']] = data_info
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        if not subband_data:
            return None, None
        
        first_data = next(iter(subband_data.values()))
        num_time_samples = first_data['time_samples']
        num_subbands = len(subband_data)
        
        sorted_subbands = sorted(subband_data.keys())
        total_channels = num_subbands * self.channels_per_subband
        
        combined_data = np.zeros((num_time_samples, total_channels))
        channel_offset = 0
        
        for sb_idx in sorted_subbands:
            sb_data = subband_data[sb_idx]['data']
            sb_channels = sb_data.shape[1]
            combined_data[:, channel_offset:channel_offset + sb_channels] = sb_data
            channel_offset += sb_channels
        
        frequencies = self._calculate_frequencies(num_subbands)
        
        # Median subtraction per channel
        ch_medians = np.median(combined_data, axis=0)
        combined_data -= ch_medians
        
        metadata = {
            'antenna': antenna,
            'time_samples': num_time_samples,
            'total_channels': total_channels,
            'num_subbands': num_subbands,
            'frequencies': frequencies,
            'start_time': first_data.get('start_time', 0.0),
            'sample_time': first_data.get('sample_time', self.time_step),
            'times': first_data.get('start_time', 0.0) + np.arange(num_time_samples) * first_data.get('sample_time', self.time_step)
        }
        
        return combined_data, metadata
    
    def load_all_data(self, antennas=None):
        """Load data for all specified antennas"""
        if antennas is None:
            antennas = self.get_available_antennas()
        
        print(f"Loading {len(antennas)} antennas...")
        
        self.all_data = {}
        self.all_meta = {}
        successful_loads = []
        
        for ant in antennas:
            data, metadata = self._load_antenna_data(ant)
            if data is not None:
                self.all_data[ant] = data
                self.all_meta[ant] = metadata
                successful_loads.append(ant)
        
        self._data_loaded = True
        
        if successful_loads:
            print(f"Successfully loaded {len(successful_loads)} antennas: {successful_loads[0]} to {successful_loads[-1]}")
        else:
            print("No antennas loaded successfully")
        
        return self.all_data, self.all_meta
    
    def get_available_antennas(self):
        """Discover available antenna numbers by scanning files"""
        antenna_set = set()
        patterns = ["*ant*subband*.spec", "ant_*_subband_*.spec"]
        
        for pattern in patterns:
            files = glob(os.path.join(self.directory, pattern))
            for filepath in files:
                filename = os.path.basename(filepath)
                matches = re.findall(r'ant[_]?(\d+)', filename)
                if matches:
                    antenna_set.add(int(matches[0]))
        
        return sorted(list(antenna_set))
    
    def compute_incoherent_sum(self, freq_bin=1, time_bin=1):
        """Compute incoherent sum of unflagged antennas"""
        if not self._data_loaded:
            raise RuntimeError("Must call load_all_data() first!")
        
        unflagged_ants = self.get_unflagged_antennas()
        if not unflagged_ants:
            raise RuntimeError("No unflagged antennas available for summation!")
        
        print(f"Computing incoherent sum of {len(unflagged_ants)} unflagged antennas")
        
        all_data = []
        ref_ant = unflagged_ants[0]
        ref_meta = self.all_meta[ref_ant]
        
        total_ants = len(unflagged_ants)
        bar = _maybe_tqdm(total_ants, desc='Summing antennas')
        for idx, ant in enumerate(unflagged_ants, start=1):
            ant_data = self.all_data[ant].copy()
            
            # Apply frequency binning
            if freq_bin > 1:
                nchan_binned = (ant_data.shape[1] // freq_bin) * freq_bin
                ant_data = ant_data[:, :nchan_binned]
                ant_data = ant_data.reshape(ant_data.shape[0], -1, freq_bin)
                ant_data = np.sum(ant_data, axis=2)
            
            # Apply time binning
            if time_bin > 1:
                nsamp_binned = (ant_data.shape[0] // time_bin) * time_bin
                ant_data = ant_data[:nsamp_binned, :]
                ant_data = ant_data.reshape(-1, time_bin, ant_data.shape[1])
                ant_data = np.sum(ant_data, axis=1)
            
            all_data.append(ant_data)
            # update progress
            if bar is not None:
                bar.update(1)
            else:
                _console_progress(idx, total_ants, prefix='Summing antennas')
        if bar is not None:
            bar.close()
        
        # Incoherent sum
        self.summed_data = np.sum(np.stack(all_data, axis=0), axis=0)
        
        # Remove channel-wise median
        ch_medians = np.median(self.summed_data, axis=0)
        self.summed_data -= ch_medians
        
        # Create metadata
        frequencies = ref_meta['frequencies']
        if freq_bin > 1:
            nchan_binned = (len(frequencies) // freq_bin) * freq_bin
            frequencies = frequencies[:nchan_binned].reshape(-1, freq_bin).mean(axis=1)
        
        times = ref_meta['times']
        if time_bin > 1:
            nsamp_binned = (len(times) // time_bin) * time_bin
            times = times[:nsamp_binned].reshape(-1, time_bin).mean(axis=1)
        
        self.sum_metadata = {
            'num_antennas': len(unflagged_ants),
            'antennas': unflagged_ants,
            'flagged_antennas': sorted(self.ant_flagged),
            'freq_bin': freq_bin,
            'time_bin': time_bin,
            'frequencies': frequencies,
            'times': times,
            'shape': self.summed_data.shape
        }
        
        return self.summed_data, self.sum_metadata
    
    def plot_waterfall(self, antenna, freq_bin=1, figsize=(9, 9)):
        """Plot waterfall for given antenna with marginal profiles"""
        if self._data_loaded and antenna in self.all_data:
            combined_data = self.all_data[antenna].copy()
            metadata = self.all_meta[antenna]
        else:
            combined_data, metadata = self._load_antenna_data(antenna)
            if combined_data is None:
                print(f"No data available for antenna {antenna}")
                return None
        
        frequencies = metadata['frequencies']
        times = metadata['times']
        
        # Apply frequency binning
        if freq_bin > 1:
            total_channels = (combined_data.shape[1] // freq_bin) * freq_bin
            combined_data_binned = combined_data[:, :total_channels]
            combined_data_binned = combined_data_binned.reshape(combined_data.shape[0], -1, freq_bin)
            combined_data = np.sum(combined_data_binned, axis=2)
            frequencies = frequencies[:total_channels].reshape(-1, freq_bin).mean(axis=1)
        
        # Calculate marginal profiles
        freq_profile = np.mean(combined_data, axis=0)
        time_profile = np.mean(combined_data, axis=1)
        snr, _, _, p = calculate_snr(time_profile)
        
        # Create plot
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05,
                              height_ratios=[1, 3, 3, 3], width_ratios=[3, 3, 3, 1])
        
        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_time = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_freq = fig.add_subplot(gs[1:, -1], sharey=ax_main)
        
        # Color scaling
        vmin = np.percentile(combined_data, 1)
        vmax = np.percentile(combined_data, 99)
        
        # Main waterfall
        im = ax_main.imshow(combined_data.T, aspect='auto', origin='lower',
                            extent=[times[0], times[-1], frequencies[-1], frequencies[0]],
                            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        
        ax_main.set_xlabel('Time (seconds)')
        ax_main.set_ylabel('Frequency (MHz)')
        
        flag_status = " (FLAGGED)" if antenna in self.ant_flagged else ""
        fig.suptitle(f'DSA110 Antenna {antenna}{flag_status}\n'
                     f'{len(frequencies)} channels, {len(times)} time samples, max SNR = {np.max(snr):.1f}')
        
        # Time profile
        color = 'red' if antenna in self.ant_flagged else 'black'
        ax_time.step(times, snr, color, where='mid', linewidth=1)
        ax_time.set_ylabel('SNR')
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(labelbottom=False)

        # Add pulse visualizations
        for start_idx, stop_idx in p:
            start_time = times[start_idx]
            stop_time = times[stop_idx]
            ax_time.axvline(start_time, color='black', linestyle='--', alpha=0.5)
            ax_time.axvline(stop_time, color='black', linestyle='--', alpha=0.5)
            ax_time.axvspan(start_time, stop_time, color='grey', alpha=0.1)
        
        # Frequency profile
        ax_freq.step(freq_profile[::-1], frequencies, color, where='mid', linewidth=1)
        ax_freq.set_xlabel('arb')
        ax_freq.yaxis.set_label_position('right')
        ax_freq.yaxis.tick_right()
        ax_freq.grid(True, alpha=0.3)
        ax_freq.tick_params(labelleft=False)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=[ax_main, ax_time, ax_freq],
                            orientation='horizontal', pad=0.1, aspect=60, fraction=0.04)
        cbar.set_label('Stokes I Power (arb. units)')
        
        plt.show()
        return fig
    
    def plot_incoherent_sum(self, freq_bin=1, time_bin=1, figsize=(9, 9)):
        """Plot the incoherent sum waterfall"""
        if self.summed_data is None:
            self.compute_incoherent_sum(freq_bin, time_bin)
        elif (self.sum_metadata['freq_bin'] != freq_bin or
              self.sum_metadata['time_bin'] != time_bin):
            # Recompute if binning parameters have changed
            self.compute_incoherent_sum(freq_bin, time_bin)
        
        data_to_plot = self.summed_data
        metadata = self.sum_metadata
        
        frequencies = metadata['frequencies']
        times = metadata['times']
        
        # Calculate marginal profiles
        freq_profile = np.mean(data_to_plot, axis=0)
        time_profile = np.mean(data_to_plot, axis=1)
        snr, _, _, p = calculate_snr(time_profile)
        
        # Create plot
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05,
                              height_ratios=[1, 3, 3, 3], width_ratios=[3, 3, 3, 1])
        
        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_time = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_freq = fig.add_subplot(gs[1:, -1], sharey=ax_main)
        
        # Color scaling
        vmin = np.percentile(data_to_plot, 1)
        vmax = np.percentile(data_to_plot, 99)
        
        # Main waterfall
        im = ax_main.imshow(data_to_plot.T, aspect='auto', origin='lower',
                            extent=[times[0], times[-1], frequencies[-1], frequencies[0]],
                            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        
        ax_main.set_xlabel('Time (seconds)')
        ax_main.set_ylabel('Frequency (MHz)')
        
        fig.suptitle(f'DSA110 Incoherent Sum of {metadata["num_antennas"]} antennas\n'
                     f'{len(frequencies)} channels, {len(times)} time samples, max SNR = {np.max(snr):.1f}')
        
        # Time profile
        ax_time.step(times, snr, 'k', where='mid', linewidth=1)
        ax_time.set_ylabel('SNR')
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(labelbottom=False)

        # Add pulse visualizations
        for start_idx, stop_idx in p:
            start_time = times[start_idx]
            stop_time = times[stop_idx]
            ax_time.axvline(start_time, color='black', linestyle='--', alpha=0.5)
            ax_time.axvline(stop_time, color='black', linestyle='--', alpha=0.5)
            ax_time.axvspan(start_time, stop_time, color='grey', alpha=0.1)
        
        # Frequency profile
        ax_freq.step(freq_profile[::-1], frequencies, 'k', where='mid', linewidth=1)
        ax_freq.set_xlabel('arb')
        ax_freq.yaxis.set_label_position('right')
        ax_freq.yaxis.tick_right()
        ax_freq.grid(True, alpha=0.3)
        ax_freq.tick_params(labelleft=False)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=[ax_main, ax_time, ax_freq],
                            orientation='horizontal', pad=0.1, aspect=60, fraction=0.04)
        cbar.set_label('Stokes I Power (arb. units)')
        
        plt.show()
        return fig
    
    def get_antenna_list(self):
        """Get list of available antenna numbers"""
        return self.get_available_antennas()
    
    def print_summary(self):
        """Print a summary of loaded data and current flags"""
        all_ants = self.get_available_antennas()
        flagged_ants = self.get_flagged_antennas()
        unflagged_ants = self.get_unflagged_antennas()
        
        print("Data Summary:")
        print(f"  Directory: {self.directory}")
        print(f"  Total antennas available: {len(all_ants)}")
        print(f"  Available antennas: {all_ants}")
        print(f"  Flagged antennas: {flagged_ants} ({len(flagged_ants)} total)")
        print(f"  Unflagged antennas: {unflagged_ants} ({len(unflagged_ants)} total)")
        
        if self._data_loaded and self.all_meta:
            meta = list(self.all_meta.values())[0]
            print(f"  Time samples: {meta['time_samples']}")
            print(f"  Total channels: {meta['total_channels']}")
            print(f"  Number of subbands: {meta['num_subbands']}")
            print(f"  Total observation time: {meta['time_samples'] * meta['sample_time']:.2f} s")



class RawVoltages(BaseAntennaProcessor):
    """Memory-efficient DSA110 raw voltage waterfall plotter with lazy loading"""

    def __init__(self, directory, ant_flagged=None, enable_caching=True):
        super().__init__(directory, ant_flagged)

        # DSA110 frequency parameters (same as waterfall plotter)
        self.subband_start_freqs = [
            1498.75, 1487.03125, 1475.3125, 1463.59375, 1451.875, 1440.15625,
            1428.4375, 1416.71875, 1405.0, 1393.28125, 1381.5625, 1369.84375,
            1358.125, 1346.40625, 1334.6875, 1322.96875
        ]
        self.channel_freq_step = -0.030517578125
        self.channels_per_subband = 384
        self.time_step = 32.768e-6  # 32.768 microseconds per sample
        
        # Performance optimizations
        self.enable_caching = enable_caching
        self._header_cache = {}  # Cache headers to avoid re-reading
        
        # NEW: Voltage data container
        self.voltage_data = {}  # antenna -> {'pol_x': array, 'pol_y': array}
        self.voltage_metadata = {}  # metadata for loaded voltage data
        self._voltage_data_loaded = False

    def _find_antenna_files(self, antenna):
        """Find all raw voltage files for given antenna"""
        patterns = [
            f"ant_{antenna:02d}_subband_*_pol_*.raw",
            f"ant_{antenna}_subband_*_pol_*.raw",
            f"*ant{antenna:02d}*subband*pol*.raw",
            f"*ant{antenna}_*subband*pol*.raw",
        ]

        files = []
        for pattern in patterns:
            found = glob(os.path.join(self.directory, pattern))
            files.extend(found)
            if found:
                break

        return sorted(list(set(files)))

    def _read_raw_voltage_file_header(self, filename):
        """Read just the header from a raw voltage file with caching"""
        if self.enable_caching and filename in self._header_cache:
            return self._header_cache[filename]
            
        with open(filename, 'rb') as f:
            header_data = f.read(24)
            if len(header_data) != 24:
                raise ValueError(f"Invalid header size in {filename}: {len(header_data)} bytes")

            header = struct.unpack('<6I', header_data)
            num_samples, num_channels, antenna_idx, subband_idx, start_time_us, sample_time_ns = header

            header_info = {
                'num_samples': num_samples,
                'num_channels': num_channels,
                'antenna': antenna_idx,
                'subband': subband_idx,
                'start_time': start_time_us / 1e6,
                'sample_time': sample_time_ns / 1e9
            }
            
            if self.enable_caching:
                self._header_cache[filename] = header_info
                
            return header_info

    def _read_raw_voltage_file(self, filename, use_mmap=False):
        """Read a single raw voltage file with optional memory mapping for large files"""
        header_info = self._read_raw_voltage_file_header(filename)
        
        if use_mmap:
            # Memory-mapped file for very large datasets
            with open(filename, 'rb') as f:
                f.seek(24)  # Skip header
                expected_floats = header_info['num_samples'] * header_info['num_channels'] * 2
                
                # Memory map the data portion
                data_array = np.memmap(f, dtype=np.float32, mode='r', 
                                     offset=24, shape=(expected_floats,))
                
                # Reshape to complex (view, not copy)
                voltage_reshaped = data_array.reshape(header_info['num_samples'], 
                                                    header_info['num_channels'], 2)
                data = voltage_reshaped[:, :, 0] + 1j * voltage_reshaped[:, :, 1]
        else:
            # Standard loading for smaller files
            with open(filename, 'rb') as f:
                # Skip header
                f.seek(24)
                
                # Read complex voltage data (real, imag interleaved)
                expected_floats = header_info['num_samples'] * header_info['num_channels'] * 2
                data_bytes = f.read(expected_floats * 4)

                if len(data_bytes) != expected_floats * 4:
                    raise ValueError(f"Data size mismatch in {filename}")

                # Convert to numpy array and reshape to complex
                voltage_array = np.frombuffer(data_bytes, dtype=np.float32)
                voltage_reshaped = voltage_array.reshape(header_info['num_samples'], header_info['num_channels'], 2)
                data = voltage_reshaped[:, :, 0] + 1j * voltage_reshaped[:, :, 1]

        header_info['data'] = data
        return header_info

    def _calculate_frequencies(self, num_subbands):
        """Calculate frequency array for all subbands"""
        total_channels = num_subbands * self.channels_per_subband
        frequencies = np.zeros(total_channels)

        for sb in range(num_subbands):
            start_freq = self.subband_start_freqs[sb]
            start_idx = sb * self.channels_per_subband

            for ch in range(self.channels_per_subband):
                frequencies[start_idx + ch] = start_freq + (ch * self.channel_freq_step)

        return frequencies

    def load_all_data(self, antennas=None):
        """Scan for available antennas and metadata without loading actual data"""
        if antennas is None:
            antennas = self.get_available_antennas()

        print(f"Scanning {len(antennas)} antennas for metadata...")

        self.all_data = {}  # Will remain empty - data loaded on demand
        self.all_meta = {}
        successful_scans = []

        for ant in antennas:
            metadata = self._get_antenna_metadata(ant)
            if metadata is not None:
                self.all_meta[ant] = metadata
                successful_scans.append(ant)

        self._data_loaded = True

        if successful_scans:
            print(f"Successfully scanned {len(successful_scans)} antennas: {successful_scans[0]} to {successful_scans[-1]}")
            # Print memory estimate per antenna
            if successful_scans:
                meta = list(self.all_meta.values())[0]
                per_ant_memory = meta['time_samples'] * meta['total_channels'] * 8 * 2 / (1024**3)  # GB for complex data
                total_memory = len(successful_scans) * per_ant_memory
        else:
            print("No antennas found")

        return self.all_data, self.all_meta

    def _get_antenna_metadata(self, antenna):
        """Get metadata for an antenna without loading the actual data"""
        files = self._find_antenna_files(antenna)
        if not files:
            return None

        # Group files by polarization
        pol_x_files = [f for f in files if '_pol_x.raw' in f]
        pol_y_files = [f for f in files if '_pol_y.raw' in f]

        if not pol_x_files:
            return None

        # Read header from first X pol file to get basic info
        try:
            header_info = self._read_raw_voltage_file_header(pol_x_files[0])
        except Exception as e:
            print(f"Error reading header from {pol_x_files[0]}: {e}")
            return None

        # Count available subbands
        x_subbands = set()
        y_subbands = set()
        
        for f in pol_x_files:
            match = re.search(r'subband_(\d+)', f)
            if match:
                x_subbands.add(int(match.group(1)))
        
        for f in pol_y_files:
            match = re.search(r'subband_(\d+)', f)
            if match:
                y_subbands.add(int(match.group(1)))

        num_subbands = len(x_subbands)
        total_channels = num_subbands * self.channels_per_subband

        frequencies = self._calculate_frequencies(num_subbands)
        times = header_info['start_time'] + np.arange(header_info['num_samples']) * header_info['sample_time']

        metadata = {
            'antenna': antenna,
            'time_samples': header_info['num_samples'],
            'total_channels': total_channels,
            'num_subbands': num_subbands,
            'frequencies': frequencies,
            'times': times,
            'start_time': header_info['start_time'],
            'sample_time': header_info['sample_time'],
            'has_both_pols': len(y_subbands) > 0,
            'pol_x_files': sorted(pol_x_files),
            'pol_y_files': sorted(pol_y_files) if pol_y_files else []
        }

        return metadata

    def _load_antenna_data_on_demand(self, antenna):
        """Load actual data for an antenna only when needed"""
        if antenna not in self.all_meta:
            print(f"No metadata available for antenna {antenna}")
            return None

        metadata = self.all_meta[antenna]
        
        # Load X polarization
        pol_x_data = self._load_polarization_data(metadata['pol_x_files'])
        if pol_x_data is None:
            return None

        # Load Y polarization if available
        pol_y_data = None
        if metadata['has_both_pols'] and metadata['pol_y_files']:
            pol_y_data = self._load_polarization_data(metadata['pol_y_files'])

        combined_data = {
            'pol_x': pol_x_data,
            'pol_y': pol_y_data,
            'has_both_pols': pol_y_data is not None
        }

        return combined_data

    def _load_polarization_data(self, file_list):
        """Load and combine data from multiple subband files for one polarization (optimized)"""
        if not file_list:
            return None

        # Use ThreadPoolExecutor for parallel file reading
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing as mp
        
        subband_data = {}
        
        # Parallel file reading
        max_workers = min(len(file_list), mp.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._read_raw_voltage_file, filename): filename 
                            for filename in file_list}
            
            for future in future_to_file:
                filename = future_to_file[future]
                try:
                    data_info = future.result()
                    subband_data[data_info['subband']] = data_info
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue

        if not subband_data:
            return None

        # Get dimensions from first file
        first_data = next(iter(subband_data.values()))
        num_time_samples = first_data['num_samples']
        sorted_subbands = sorted(subband_data.keys())
        num_subbands = len(sorted_subbands)
        total_channels = num_subbands * self.channels_per_subband

        # Pre-allocate combined array
        combined_data = np.zeros((num_time_samples, total_channels), dtype=np.complex64)
        
        # Vectorized combination
        for i, sb_idx in enumerate(sorted_subbands):
            if sb_idx in subband_data:
                start_ch = i * self.channels_per_subband
                end_ch = start_ch + self.channels_per_subband
                combined_data[:, start_ch:end_ch] = subband_data[sb_idx]['data']

        return combined_data

    def get_available_antennas(self):
        """Discover available antenna numbers by scanning raw voltage files"""
        antenna_set = set()
        patterns = ["*ant*subband*pol*.raw", "ant_*_subband_*_pol_*.raw"]

        for pattern in patterns:
            files = glob(os.path.join(self.directory, pattern))
            for filepath in files:
                filename = os.path.basename(filepath)
                matches = re.findall(r'ant[_]?(\d+)', filename)
                if matches:
                    antenna_set.add(int(matches[0]))

        return sorted(list(antenna_set))

    def load_voltage_data(self, antennas=None, start_time=None, stop_time=None, start_sample=None, stop_sample=None):
        """
        Load actual complex voltage data into memory for specified antennas.
        
        Parameters
        ----------
        antennas : list or None
            List of antenna numbers to load. If None, loads all unflagged antennas.
        start_time : float or None
            Start time in seconds for time slicing
        stop_time : float or None
            Stop time in seconds for time slicing
        start_sample : int or None
            Start sample index for time slicing (overrides start_time)
        stop_sample : int or None
            Stop sample index for time slicing (overrides stop_time)
        
        Returns
        -------
        dict
            Dictionary mapping antenna number to voltage data structure
        """
        if not self._data_loaded:
            raise RuntimeError("Must call load_all_data() first to scan metadata!")
        
        if antennas is None:
            antennas = self.get_unflagged_antennas()
        
        print(f"Loading voltage data for {len(antennas)} antennas...")
        
        self.voltage_data = {}
        successful_loads = 0
        
        total_ants = len(antennas)
        bar = _maybe_tqdm(total_ants, desc='Loading voltage data')
        
        for i, ant in enumerate(antennas):
            if ant not in self.all_meta:
                print(f"Warning: No metadata for antenna {ant}, skipping")
                continue
            
            # Load antenna data
            ant_data = self._load_antenna_data_on_demand(ant)
            if ant_data is None:
                print(f"Warning: Could not load data for antenna {ant}, skipping")
                continue
            
            # Apply time slicing if requested
            pol_x = ant_data['pol_x']
            pol_y = ant_data['pol_y'] if ant_data['has_both_pols'] else None
            
            # Determine time slice indices
            if start_sample is not None or stop_sample is not None:
                # Use sample indices directly
                start_idx = start_sample if start_sample is not None else 0
                end_idx = stop_sample if stop_sample is not None else pol_x.shape[0]
            elif start_time is not None or stop_time is not None:
                # Convert times to sample indices
                metadata = self.all_meta[ant]
                sample_time = metadata['sample_time']
                start_time_offset = metadata['start_time']
                
                if start_time is not None:
                    start_idx = int((start_time - start_time_offset) / sample_time)
                else:
                    start_idx = 0
                    
                if stop_time is not None:
                    end_idx = int((stop_time - start_time_offset) / sample_time)
                else:
                    end_idx = pol_x.shape[0]
            else:
                # No time slicing
                start_idx = 0
                end_idx = pol_x.shape[0]
            
            # Apply slicing
            pol_x = pol_x[start_idx:end_idx, :]
            if pol_y is not None:
                pol_y = pol_y[start_idx:end_idx, :]
            
            # Store in structured format
            self.voltage_data[ant] = {
                'pol_x': pol_x,
                'pol_y': pol_y,
                'has_both_pols': ant_data['has_both_pols'],
                'antenna': ant,
                'time_slice': (start_idx, end_idx)
            }
            
            successful_loads += 1
            
            if bar is not None:
                bar.update(1)
            else:
                _console_progress(i + 1, total_ants, prefix='Loading voltage data')
        
        if bar is not None:
            bar.close()
        
        self._voltage_data_loaded = True
        
        # Store metadata for loaded data
        if successful_loads > 0:
            ref_ant = list(self.voltage_data.keys())[0]
            ref_meta = self.all_meta[ref_ant]
            sample_data = self.voltage_data[ref_ant]
            
            # Calculate time array for the loaded slice
            sample_time = ref_meta['sample_time']
            start_time_offset = ref_meta['start_time']
            start_idx, end_idx = sample_data['time_slice']
            times = start_time_offset + np.arange(start_idx, end_idx) * sample_time
            
            self.voltage_metadata = {
                'loaded_antennas': list(self.voltage_data.keys()),
                'num_antennas': successful_loads,
                'time_slice_samples': (start_idx, end_idx),
                'time_slice_seconds': (times[0], times[-1]) if len(times) > 0 else (0, 0),
                'times': times,
                'frequencies': ref_meta['frequencies'],
                'total_channels': ref_meta['total_channels'],
                'num_time_samples': sample_data['pol_x'].shape[0],
                'sample_time': sample_time
            }
        
        print(f"Successfully loaded voltage data for {successful_loads} antennas")
        if successful_loads > 0:
            sample_shape = sample_data['pol_x'].shape
            print(f"Data shape per antenna: {sample_shape}")
            total_gb = successful_loads * sample_data['pol_x'].nbytes * 2 / (1024**3)  # *2 for both pols
            print(f"Total memory usage: {total_gb:.2f} GB")
            print(f"Time range: {self.voltage_metadata['time_slice_seconds'][0]:.6f} to {self.voltage_metadata['time_slice_seconds'][1]:.6f} seconds")
        
        return self.voltage_data

    def get_antenna_voltages(self, antenna):
        """
        Get voltage data for a specific antenna.
        
        Parameters
        ----------
        antenna : int
            Antenna number
            
        Returns
        -------
        dict
            Structured data with 'pol_x', 'pol_y', 'frequencies', 'times'
        """
        if not self._voltage_data_loaded:
            raise RuntimeError("Must call load_voltage_data() first!")
        
        if antenna not in self.voltage_data:
            raise ValueError(f"Antenna {antenna} not loaded. Available: {list(self.voltage_data.keys())}")
        
        ant_data = self.voltage_data[antenna]
        
        return {
            'pol_x': ant_data['pol_x'],
            'pol_y': ant_data['pol_y'],
            'has_both_pols': ant_data['has_both_pols'],
            'antenna': antenna,
            'frequencies': self.voltage_metadata['frequencies'],
            'times': self.voltage_metadata['times'],
            'shape': ant_data['pol_x'].shape,
            'time_slice': ant_data['time_slice']
        }

    def transpose_voltages(self, freq_channel):
        """
        Transpose voltage data to get all antennas for a specific frequency channel.
        
        Parameters
        ----------
        freq_channel : int
            Frequency channel index (0 to total_channels-1)
            
        Returns
        -------
        dict
            Structured data with voltage time series for all antennas at given frequency:
            {
                'pol_x': array of shape (num_time_samples, num_antennas),
                'pol_y': array of shape (num_time_samples, num_antennas),
                'antennas': list of antenna numbers,
                'frequency': frequency in MHz,
                'times': time array,
                'freq_channel': channel index
            }
        """
        if not self._voltage_data_loaded:
            raise RuntimeError("Must call load_voltage_data() first!")
        
        if not self.voltage_data:
            raise RuntimeError("No voltage data loaded!")
        
        total_channels = self.voltage_metadata['total_channels']
        if freq_channel < 0 or freq_channel >= total_channels:
            raise ValueError(f"freq_channel must be between 0 and {total_channels-1}")
        
        antennas = sorted(self.voltage_data.keys())
        num_antennas = len(antennas)
        num_time_samples = self.voltage_metadata['num_time_samples']
        
        # Pre-allocate arrays
        pol_x_transposed = np.zeros((num_time_samples, num_antennas), dtype=np.complex64)
        pol_y_transposed = np.zeros((num_time_samples, num_antennas), dtype=np.complex64)
        
        # Fill arrays with voltage data for the specified frequency
        for i, ant in enumerate(antennas):
            ant_data = self.voltage_data[ant]
            pol_x_transposed[:, i] = ant_data['pol_x'][:, freq_channel]
            
            if ant_data['has_both_pols'] and ant_data['pol_y'] is not None:
                pol_y_transposed[:, i] = ant_data['pol_y'][:, freq_channel]
            else:
                # Fill with zeros if no Y pol data
                pol_y_transposed[:, i] = 0.0
        
        frequency_mhz = self.voltage_metadata['frequencies'][freq_channel]
        
        return {
            'pol_x': pol_x_transposed,
            'pol_y': pol_y_transposed,
            'antennas': antennas,
            'frequency': frequency_mhz,
            'freq_channel': freq_channel,
            'times': self.voltage_metadata['times'],
            'shape': pol_x_transposed.shape
        }

    def compute_n2_matrix(self, freq_channels=None, polarization='x', sum_time=True):
        """
        Compute N2 visibility matrix (outer product of voltage streams) for given frequencies.
        
        The N2 matrix is computed as: N2[i,j] = V_i * conj(V_j)
        where V_i and V_j are voltage streams from antennas i and j.
        
        Parameters
        ----------
        freq_channels : list or int or None
            Frequency channel indices. If None, uses all channels.
            If int, uses single channel. If list, uses specified channels.
        polarization : str
            Polarization to use ('x' or 'y')
        sum_time : bool
            If True, sum the N2 matrices over time samples.
            If False, return full time-resolved N2 matrix.
            
        Returns
        -------
        N2Matrix
            N2Matrix object containing the computed matrices with shape:
            - If sum_time=True: (num_freq_channels, num_antennas, num_antennas)
            - If sum_time=False: (num_time_samples, num_freq_channels, num_antennas, num_antennas)
        """
        if not self._voltage_data_loaded:
            raise RuntimeError("Must call load_voltage_data() first!")
        
        # Handle frequency channel specification
        total_channels = self.voltage_metadata['total_channels']
        if freq_channels is None:
            freq_channels = list(range(total_channels))
        elif isinstance(freq_channels, int):
            freq_channels = [freq_channels]
        else:
            freq_channels = list(freq_channels)
        
        # Validate frequency channels
        for fc in freq_channels:
            if fc < 0 or fc >= total_channels:
                raise ValueError(f"freq_channel {fc} must be between 0 and {total_channels-1}")
        
        antennas = sorted(self.voltage_data.keys())
        num_antennas = len(antennas)
        num_time_samples = self.voltage_metadata['num_time_samples']
        
        print(f"Computing N2 matrix for {len(freq_channels)} frequency channels...")
        
        if sum_time:
            # Sum over time samples: shape (num_freq_channels, num_antennas, num_antennas)
            n2_matrix = np.zeros((len(freq_channels), num_antennas, num_antennas), dtype=np.complex64)
            
            bar = _maybe_tqdm(len(freq_channels), desc='Computing N2 matrices')
            
            for i, freq_ch in enumerate(freq_channels):
                # Get transposed voltage data for this frequency
                voltage_data = self.transpose_voltages(freq_ch)
                
                if polarization.lower() == 'x':
                    voltages = voltage_data['pol_x']  # shape: (num_time_samples, num_antennas)
                elif polarization.lower() == 'y':
                    voltages = voltage_data['pol_y']
                else:
                    raise ValueError("polarization must be 'x' or 'y'")
                
                # Compute N2 for this frequency using vectorized operations
                # voltages: (num_time_samples, num_antennas)
                # We want: n2[i, j] = sum_t(voltages[t, i] * conj(voltages[t, j]))
                voltages_expanded_i = voltages[:, :, np.newaxis]  # (num_time_samples, num_antennas, 1)
                voltages_expanded_j = voltages[:, np.newaxis, :]  # (num_time_samples, 1, num_antennas)
                
                n2_freq = voltages_expanded_i * np.conj(voltages_expanded_j)
                # Sum over time: (num_antennas, num_antennas)
                n2_matrix[i] = np.sum(n2_freq, axis=0)
                
                if bar is not None:
                    bar.update(1)
                else:
                    _console_progress(i + 1, len(freq_channels), prefix='Computing N2')
            
            if bar is not None:
                bar.close()
            
            frequencies_used = [self.voltage_metadata['frequencies'][fc] for fc in freq_channels]
            
            # Create metadata dictionary
            metadata = {
                'computation_type': 'time_summed',
                'num_frequencies': len(freq_channels),
                'original_time_samples': num_time_samples,
                'matrix_dtype': str(n2_matrix.dtype),
                'shape_description': f'({len(freq_channels)}, {num_antennas}, {num_antennas})'
            }
            
            return N2Matrix(
                n2_data=n2_matrix,
                antennas=antennas,
                frequencies=frequencies_used,
                freq_channels=freq_channels,
                polarization=polarization,
                sum_frequencies=False,  # We never sum frequencies, only time
                times=None,  # No time dimension after summing
                metadata=metadata
            )
        
        else:
            # Keep time resolution: shape (num_time_samples, num_freq_channels, num_antennas, num_antennas)
            n2_matrix = np.zeros((num_time_samples, len(freq_channels), num_antennas, num_antennas), dtype=np.complex64)
            
            bar = _maybe_tqdm(len(freq_channels), desc='Computing N2 matrices')
            
            for i, freq_ch in enumerate(freq_channels):
                # Get transposed voltage data for this frequency
                voltage_data = self.transpose_voltages(freq_ch)
                
                if polarization.lower() == 'x':
                    voltages = voltage_data['pol_x']  # shape: (num_time_samples, num_antennas)
                elif polarization.lower() == 'y':
                    voltages = voltage_data['pol_y']
                else:
                    raise ValueError("polarization must be 'x' or 'y'")
                
                # Compute N2 for this frequency using vectorized operations
                # voltages: (num_time_samples, num_antennas)
                # We want: n2[t, i, j] = voltages[t, i] * conj(voltages[t, j])
                voltages_expanded_i = voltages[:, :, np.newaxis]  # (num_time_samples, num_antennas, 1)
                voltages_expanded_j = voltages[:, np.newaxis, :]  # (num_time_samples, 1, num_antennas)
                
                n2_freq = voltages_expanded_i * np.conj(voltages_expanded_j)
                # Store in frequency dimension: (num_time_samples, num_antennas, num_antennas)
                n2_matrix[:, i, :, :] = n2_freq
                
                if bar is not None:
                    bar.update(1)
                else:
                    _console_progress(i + 1, len(freq_channels), prefix='Computing N2')
            
            if bar is not None:
                bar.close()
            
            frequencies_used = [self.voltage_metadata['frequencies'][fc] for fc in freq_channels]
            
            # Create metadata dictionary
            metadata = {
                'computation_type': 'time_resolved',
                'num_frequencies': len(freq_channels),
                'num_time_samples': num_time_samples,
                'matrix_dtype': str(n2_matrix.dtype),
                'shape_description': f'({num_time_samples}, {len(freq_channels)}, {num_antennas}, {num_antennas})'
            }
            
            return N2Matrix(
                n2_data=n2_matrix,
                antennas=antennas,
                frequencies=frequencies_used,
                freq_channels=freq_channels,
                polarization=polarization,
                sum_frequencies=False,  # We never sum frequencies
                times=self.voltage_metadata['times'],
                metadata=metadata
            )

    def compute_incoherent_sum(self, freq_bin=1, time_bin=1):
        """Compute incoherent sum of unflagged antennas using optimized lazy loading"""
        if not self._data_loaded:
            raise RuntimeError("Must call load_all_data() first!")

        unflagged_ants = self.get_unflagged_antennas()
        if not unflagged_ants:
            raise RuntimeError("No unflagged antennas available for summation!")

        print(f"Computing incoherent sum of {len(unflagged_ants)} unflagged antennas")

        # Get reference antenna for metadata
        ref_ant = unflagged_ants[0]
        ref_meta = self.all_meta[ref_ant]

        # Pre-calculate output dimensions
        orig_time_samples = ref_meta['time_samples']
        orig_channels = ref_meta['total_channels']
        
        # Calculate binned dimensions
        time_samples_binned = (orig_time_samples // time_bin) * time_bin
        channels_binned = (orig_channels // freq_bin) * freq_bin
        
        final_time_samples = time_samples_binned // time_bin
        final_channels = channels_binned // freq_bin

        # Pre-allocate sum array
        sum_stokes_i = np.zeros((final_time_samples, final_channels), dtype=np.float32)
        successful_ants = 0

        # Process antennas with optimized power calculation and binning
        total_ants = len(unflagged_ants)
        bar = _maybe_tqdm(total_ants, desc='Processing antennas')
        for i, ant in enumerate(unflagged_ants):            
            # Load data on demand
            ant_data = self._load_antenna_data_on_demand(ant)
            if ant_data is None:
                print(f"Warning: Could not load data for antenna {ant}, skipping")
                continue
            
            # Optimized power calculation using real/imag components directly
            pol_x = ant_data['pol_x']
            pol_x_power = pol_x.real**2 + pol_x.imag**2

            if ant_data['has_both_pols'] and ant_data['pol_y'] is not None:
                pol_y = ant_data['pol_y']
                pol_y_power = pol_y.real**2 + pol_y.imag**2
                stokes_i = pol_x_power + pol_y_power
            else:
                stokes_i = pol_x_power

            # Optimized binning using array reshaping
            if freq_bin > 1 or time_bin > 1:
                # Trim to binnable size
                stokes_i = stokes_i[:time_samples_binned, :channels_binned]
                
                # Apply both binnings in one step using reshape
                if freq_bin > 1 and time_bin > 1:
                    stokes_i = stokes_i.reshape(final_time_samples, time_bin, 
                                              final_channels, freq_bin)
                    stokes_i = np.sum(stokes_i, axis=(1, 3))
                elif freq_bin > 1:
                    stokes_i = stokes_i.reshape(stokes_i.shape[0], final_channels, freq_bin)
                    stokes_i = np.sum(stokes_i, axis=2)
                elif time_bin > 1:
                    stokes_i = stokes_i.reshape(final_time_samples, time_bin, stokes_i.shape[1])
                    stokes_i = np.sum(stokes_i, axis=1)

            # Add to sum (in-place operation)
            sum_stokes_i += stokes_i
            successful_ants += 1
            # update progress bar per antenna processed
            if bar is not None:
                bar.update(1)
            else:
                _console_progress(successful_ants, total_ants, prefix='Processed antennas')

            # Explicitly delete to free memory immediately
            del ant_data, pol_x, stokes_i
            if 'pol_y' in locals():
                del pol_y
        if bar is not None:
            bar.close()

        if successful_ants == 0:
            raise RuntimeError("No valid antenna data found for incoherent sum")

        self.summed_data = sum_stokes_i
        print(f"Successfully summed {successful_ants} antennas")

        # Update frequencies and times for binning (vectorized)
        frequencies = ref_meta['frequencies']
        if freq_bin > 1:
            frequencies = frequencies[:channels_binned].reshape(-1, freq_bin).mean(axis=1)

        times = ref_meta['times']
        if time_bin > 1:
            times = times[:time_samples_binned].reshape(-1, time_bin).mean(axis=1)

        self.sum_metadata = {
            'num_antennas': successful_ants,
            'antennas': unflagged_ants,
            'flagged_antennas': sorted(self.ant_flagged),
            'freq_bin': freq_bin,
            'time_bin': time_bin,
            'frequencies': frequencies,
            'times': times,
            'shape': self.summed_data.shape
        }

        return self.summed_data, self.sum_metadata


    def plot_waterfall(self, antenna, freq_bin=1, time_bin=1, figsize=(12, 12)):
        """Plot waterfall for given antenna showing power spectra for X, Y polarizations and Stokes I (optimized)"""
        # Load data on demand
        combined_data = self._load_antenna_data_on_demand(antenna)
        if combined_data is None:
            print(f"No data available for antenna {antenna}")
            return None

        metadata = self.all_meta[antenna]
        frequencies = metadata['frequencies']
        times = metadata['times']

        # Pre-calculate dimensions for binning
        orig_time_samples = combined_data['pol_x'].shape[0]
        orig_channels = combined_data['pol_x'].shape[1]
        
        time_samples_binned = (orig_time_samples // time_bin) * time_bin
        channels_binned = (orig_channels // freq_bin) * freq_bin
        
        final_time_samples = time_samples_binned // time_bin
        final_channels = channels_binned // freq_bin

        # Optimized power calculation
        pol_x = combined_data['pol_x'][:time_samples_binned, :channels_binned]
        pol_x_power = pol_x.real**2 + pol_x.imag**2
        
        pol_y_power = None
        if combined_data['has_both_pols']:
            pol_y = combined_data['pol_y'][:time_samples_binned, :channels_binned]
            pol_y_power = pol_y.real**2 + pol_y.imag**2

        # Optimized binning using reshape (same technique as compute_incoherent_sum)
        if freq_bin > 1 and time_bin > 1:
            # Both binnings in one step
            pol_x_power = pol_x_power.reshape(final_time_samples, time_bin, 
                                            final_channels, freq_bin)
            pol_x_power = np.sum(pol_x_power, axis=(1, 3))
            
            if pol_y_power is not None:
                pol_y_power = pol_y_power.reshape(final_time_samples, time_bin,
                                                final_channels, freq_bin)
                pol_y_power = np.sum(pol_y_power, axis=(1, 3))
        elif freq_bin > 1:
            pol_x_power = pol_x_power.reshape(pol_x_power.shape[0], final_channels, freq_bin)
            pol_x_power = np.sum(pol_x_power, axis=2)
            
            if pol_y_power is not None:
                pol_y_power = pol_y_power.reshape(pol_y_power.shape[0], final_channels, freq_bin)
                pol_y_power = np.sum(pol_y_power, axis=2)
        elif time_bin > 1:
            pol_x_power = pol_x_power.reshape(final_time_samples, time_bin, pol_x_power.shape[1])
            pol_x_power = np.sum(pol_x_power, axis=1)
            
            if pol_y_power is not None:
                pol_y_power = pol_y_power.reshape(final_time_samples, time_bin, pol_y_power.shape[1])
                pol_y_power = np.sum(pol_y_power, axis=1)

        # Calculate Stokes I
        stokes_i = pol_x_power + pol_y_power

        ch_median = np.median(stokes_i, axis=0)
        stokes_i -= ch_median

        # Update frequency and time arrays
        if freq_bin > 1:
            frequencies = frequencies[:channels_binned].reshape(-1, freq_bin).mean(axis=1)
        if time_bin > 1:
            times = times[:time_samples_binned].reshape(-1, time_bin).mean(axis=1)

        # Calculate marginal profiles for Stokes I (like DSA110WaterfallPlotter)
        freq_profile = np.mean(stokes_i, axis=0)
        time_profile = np.mean(stokes_i, axis=1)
        snr, _, _, p = calculate_snr(time_profile)

        # Create plot with gridspec layout (matching DSA110WaterfallPlotter style)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05,
                              height_ratios=[1, 3, 3, 3], width_ratios=[3, 3, 3, 1])

        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_time = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_freq = fig.add_subplot(gs[1:, -1], sharey=ax_main)

        # Color scaling (use Stokes I for common scale)
        vmin = np.percentile(stokes_i, 1)
        vmax = np.percentile(stokes_i, 99)

        # Main waterfall (Stokes I only, like DSA110WaterfallPlotter)
        im = ax_main.imshow(stokes_i.T, aspect='auto', origin='lower',
                           extent=[times[0], times[-1], frequencies[-1], frequencies[0]],
                           cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")

        ax_main.set_xlabel('Time (seconds)')
        ax_main.set_ylabel('Frequency (MHz)')

        # Flag status and title formatting
        flag_status = " (FLAGGED)" if antenna in self.ant_flagged else ""
        max_snr_text = f", max SNR = {np.max(snr):.1f}"
        fig.suptitle(f'DSA110 Raw Voltage Antenna {antenna}{flag_status}\n'
                     f'{len(frequencies)} channels, {len(times)} time samples{max_snr_text}')

        # Time profile (SNR plot like DSA110WaterfallPlotter)
        color = 'red' if antenna in self.ant_flagged else 'black'
        ax_time.step(times, snr, color, where='mid', linewidth=1)
        ax_time.set_ylabel('SNR')
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(labelbottom=False)

        # Add pulse visualizations (like DSA110WaterfallPlotter)
        for start_idx, stop_idx in p:
            start_time = times[start_idx]
            stop_time = times[stop_idx]
            ax_time.axvline(start_time, color='black', linestyle='--', alpha=0.5)
            ax_time.axvline(stop_time, color='black', linestyle='--', alpha=0.5)
            ax_time.axvspan(start_time, stop_time, color='grey', alpha=0.1)

        # Frequency profile (like DSA110WaterfallPlotter)
        ax_freq.step(freq_profile[::-1], frequencies, color, where='mid', linewidth=1)
        ax_freq.set_xlabel('arb')
        ax_freq.yaxis.set_label_position('right')
        ax_freq.yaxis.tick_right()
        ax_freq.grid(True, alpha=0.3)
        ax_freq.tick_params(labelleft=False)

        # Colorbar (matching DSA110WaterfallPlotter style)
        cbar = fig.colorbar(im, ax=[ax_main, ax_time, ax_freq],
                           orientation='horizontal', pad=0.1, aspect=60, fraction=0.04)
        cbar.set_label('Stokes I Power (arb. units)')

        plt.show()
        return fig

    def plot_incoherent_sum(self, freq_bin=1, time_bin=1, figsize=(9, 9)):
        """Plot the incoherent sum waterfall"""
        if self.summed_data is None:
            self.compute_incoherent_sum_(freq_bin, time_bin)
        elif (self.sum_metadata['freq_bin'] != freq_bin or
              self.sum_metadata['time_bin'] != time_bin):
            # Recompute if binning parameters have changed
            self.compute_incoherent_sum_(freq_bin, time_bin)

        data_to_plot = self.summed_data

        ch_median = np.median(data_to_plot, axis=0)
        data_to_plot -= ch_median

        metadata = self.sum_metadata

        frequencies = metadata['frequencies']
        times = metadata['times']

        # Create plot
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05,
                              height_ratios=[1, 3, 3, 3], width_ratios=[3, 3, 3, 1])

        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_time = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_freq = fig.add_subplot(gs[1:, -1], sharey=ax_main)

        # Color scaling
        vmin = np.percentile(data_to_plot, 1)
        vmax = np.percentile(data_to_plot, 99)

        # Main waterfall
        im = ax_main.imshow(data_to_plot.T, aspect='auto', origin='lower',
                            extent=[times[0], times[-1], frequencies[-1], frequencies[0]],
                            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")

        ax_main.set_xlabel('Time (seconds)')
        ax_main.set_ylabel('Frequency (MHz)')


        # Time profile (SNR-like using total power)
        time_profile = np.mean(data_to_plot, axis=1)
        snr, _, _, p = calculate_snr(time_profile)

        fig.suptitle(f'DSA110 Raw Voltage Incoherent Sum of {metadata["num_antennas"]} antennas\n'
                     f'{len(frequencies)} channels, {len(times)} time samples, Max SNR: {np.max(snr)}')


        ax_time.step(times, snr, 'k', where='mid', linewidth=1)
        ax_time.set_ylabel('SNR')
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(labelbottom=False)

        # Add pulse visualizations
        for start_idx, stop_idx in p:
            start_time = times[start_idx]
            stop_time = times[stop_idx]
            ax_time.axvline(start_time, color='black', linestyle='--', alpha=0.5)
            ax_time.axvline(stop_time, color='black', linestyle='--', alpha=0.5)
            ax_time.axvspan(start_time, stop_time, color='grey', alpha=0.1)

        # Frequency profile
        freq_profile = np.mean(data_to_plot, axis=0)
        ax_freq.step(freq_profile[::-1], frequencies, 'k', where='mid', linewidth=1)
        ax_freq.set_xlabel('arb')
        ax_freq.yaxis.set_label_position('right')
        ax_freq.yaxis.tick_right()
        ax_freq.grid(True, alpha=0.3)
        ax_freq.tick_params(labelleft=False)

        # Colorbar
        cbar = fig.colorbar(im, ax=[ax_main, ax_time, ax_freq],
                            orientation='horizontal', pad=0.1, aspect=60, fraction=0.04)
        cbar.set_label('Power (arb. units)')

        plt.show()
        return fig

    def get_antenna_list(self):
        """Get list of available antenna numbers"""
        return self.get_available_antennas()

    def print_summary(self):
        """Print a summary of loaded data and current flags"""
        all_ants = self.get_available_antennas()
        flagged_ants = self.get_flagged_antennas()
        unflagged_ants = self.get_unflagged_antennas()

        print("Raw Voltage Data Summary:")
        print(f"  Directory: {self.directory}")
        print(f"  Total antennas available: {len(all_ants)}")
        print(f"  Available antennas: {all_ants}")
        print(f"  Flagged antennas: {flagged_ants} ({len(flagged_ants)} total)")
        print(f"  Unflagged antennas: {unflagged_ants} ({len(unflagged_ants)} total)")

        if self._data_loaded and self.all_meta:
            meta = list(self.all_meta.values())[0]
            print(f"  Time samples: {meta['time_samples']}")
            print(f"  Total channels: {meta['total_channels']}")
            print(f"  Number of subbands: {meta['num_subbands']}")
            print(f"  Total observation time: {meta['time_samples'] * meta['sample_time']:.2f} s")
            per_ant_gb = meta['time_samples'] * meta['total_channels'] * 8 * 2 / (1024**3)
            print(f"  Memory per antenna: {per_ant_gb:.2f} GB (using lazy loading)")