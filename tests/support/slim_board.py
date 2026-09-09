"""slim_board.py -- PINNED CONSTANTS of the SLIM travelling-wave board.

Split out: these describe OUR board (phase count, electrode
pitch, and the wave-speed relation that follows from them).
"""
N_PHASE = 8
PITCH_MM = 1.143            # TW electrode pitch along x (1.206 - 0.064)


def wave_speed_mm_us(freq_hz):
    """Transport (phase) velocity of the traveling wave, mm/us.

    CONVENTION: `freq_hz` is the UNIFIED WAVEFORM frequency — the rate at
    which the whole N_PHASE-pad pattern repeats. The pattern advances one full
    wavelength (N_PHASE pitches) per waveform period, so

        v_wave = N_PHASE * PITCH_MM * f_waveform.
"""
    return N_PHASE * PITCH_MM * freq_hz * 1e-6
