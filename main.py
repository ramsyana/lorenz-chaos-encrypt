import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from chaos_encryption.lorenz import solve_lorenz, solve_lorenz_receiver
from chaos_encryption.messages import generate_monochromatic, text_to_integers, integers_to_signal, signal_to_integers, integers_to_text
from chaos_encryption.synchronization import sync_encrypt, sync_decrypt
from chaos_encryption.pce import pce_add_encrypt, pce_add_decrypt, pce_convolve_encrypt, pce_convolve_decrypt
from chaos_encryption.hacking import hack_sync_signal, hack_pce_signal
from chaos_encryption.utils import calculate_l2_norm, calculate_fft, calculate_fidelity_text

# Parameters
a, b, r = 10, 8/3, 28  # Lorenz parameters
u0 = [5.0, 5.0, 5.0]   # Emitter ICs
ur0 = [25.0, 6.0, 50.0] # Receiver ICs
t_span = [0, 200]       # Time span
dt = 0.001             # Time step
A = 1                  # Message amplitude
omega = 10             # Message frequency (also test with 80)
t0 = 100               # Message center time
w_co = 35              # Hacking cutoff frequency

# Generate time vector
t = np.arange(t_span[0], t_span[1], dt)

# Generate Lorenz signal (emitter)
t_lorenz, sol = solve_lorenz(u0, t_span, dt, a, b, r)
u_signal = sol[0]  # u component

# Generate message signals
# 1. Monochromatic signal
m_mono = generate_monochromatic(t, A, omega, t0)

# 2. Text signal
text = "Hello World!"
m_discrete = text_to_integers(text)
m_text = integers_to_signal(m_discrete, t, t0, n_steps=100)

def analyze_encryption_method(name, u_signal, m_signal, encryption_fn, decryption_fn, hacking_fn):
    """Analyze an encryption method with given signals and functions."""
    # Encrypt
    me_signal = encryption_fn(u_signal, m_signal)
    
    # Decrypt
    mr_signal = decryption_fn(me_signal, u_signal)
    
    # Hack
    mh_signal = hacking_fn(me_signal, dt, w_co)
    
    # Calculate errors
    error_decrypt = calculate_l2_norm(m_signal, mr_signal)
    error_hack = calculate_l2_norm(m_signal, mh_signal)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.plot(t, m_signal, label='Original')
    plt.title(f'{name}: Original Message')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(t, mr_signal, label='Recovered')
    plt.title(f'Recovered Message (L2 error: {error_decrypt:.6f})')
    plt.legend()
    
    plt.subplot(313)
    plt.plot(t, mh_signal, label='Hacked')
    plt.title(f'Hacked Message (L2 error: {error_hack:.6f})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{name.lower().replace(" ", "_")}_analysis.png')
    plt.close()
    
    return error_decrypt, error_hack

# Analyze synchronization method
def analyze_sync(u_signal, m_signal):
    # Encrypt
    me_signal = sync_encrypt(u_signal, m_signal)
    
    # Simulate receiver
    t_rec, sol_r = solve_lorenz_receiver(ur0, t, me_signal, t, a, b, r)
    ur_signal = sol_r[0]
    
    # Decrypt and hack
    mr_signal = sync_decrypt(me_signal, ur_signal)
    mh_signal = hack_sync_signal(ur_signal, dt, w_co)
    
    # Calculate errors
    error_decrypt = calculate_l2_norm(m_signal, mr_signal)
    error_hack = calculate_l2_norm(m_signal, mh_signal)
    
    return error_decrypt, error_hack

# Run analysis for monochromatic signal
print("\nAnalyzing monochromatic signal...")

# 1. Synchronization
sync_errors = analyze_sync(u_signal, m_mono)
print(f"Synchronization - Decrypt Error: {sync_errors[0]:.6f}, Hack Error: {sync_errors[1]:.6f}")

# 2. PCE Addition
pce_add_errors = analyze_encryption_method(
    "PCE Addition",
    u_signal,
    m_mono,
    pce_add_encrypt,
    pce_add_decrypt,
    hack_pce_signal
)
print(f"PCE Addition - Decrypt Error: {pce_add_errors[0]:.6f}, Hack Error: {pce_add_errors[1]:.6f}")

# 3. PCE Convolution
pce_conv_errors = analyze_encryption_method(
    "PCE Convolution",
    u_signal,
    m_mono,
    pce_convolve_encrypt,
    pce_convolve_decrypt,
    hack_pce_signal
)
print(f"PCE Convolution - Decrypt Error: {pce_conv_errors[0]:.6f}, Hack Error: {pce_conv_errors[1]:.6f}")

# Run analysis for text signal
print("\nAnalyzing text signal...")

# 1. Synchronization
sync_text_errors = analyze_sync(u_signal, m_text)
print(f"Synchronization - Decrypt Error: {sync_text_errors[0]:.6f}, Hack Error: {sync_text_errors[1]:.6f}")

# 2. PCE Addition
pce_add_text_errors = analyze_encryption_method(
    "PCE Addition Text",
    u_signal,
    m_text,
    pce_add_encrypt,
    pce_add_decrypt,
    hack_pce_signal
)
print(f"PCE Addition - Decrypt Error: {pce_add_text_errors[0]:.6f}, Hack Error: {pce_add_text_errors[1]:.6f}")

# 3. PCE Convolution
pce_conv_text_errors = analyze_encryption_method(
    "PCE Convolution Text",
    u_signal,
    m_text,
    pce_convolve_encrypt,
    pce_convolve_decrypt,
    hack_pce_signal
)
print(f"PCE Convolution - Decrypt Error: {pce_conv_text_errors[0]:.6f}, Hack Error: {pce_conv_text_errors[1]:.6f}")

# Recover and check text fidelity
def analyze_text_recovery(name, encryption_fn, decryption_fn):
    me_signal = encryption_fn(u_signal, m_text)
    mr_signal = decryption_fn(me_signal, u_signal)
    
    # Convert recovered signal back to text
    m_rec_discrete = signal_to_integers(mr_signal, t, t0, n_steps=100)
    recovered_text = integers_to_text(m_rec_discrete)
    
    # Calculate fidelity
    fidelity = calculate_fidelity_text(text, recovered_text)
    print(f"\n{name} Text Recovery:")
    print(f"Original text: {text}")
    print(f"Recovered text: {recovered_text}")
    print(f"Fidelity: {fidelity:.2f}%")

# Analyze text recovery for each method
analyze_text_recovery("PCE Addition", pce_add_encrypt, pce_add_decrypt)
analyze_text_recovery("PCE Convolution", pce_convolve_encrypt, pce_convolve_decrypt)

# Plot frequency analysis
def plot_frequency_analysis(signal, name):
    freqs, fft = calculate_fft(signal, dt)
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, np.abs(fft))
    plt.title(f'Frequency Spectrum - {name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.savefig(f'{name.lower().replace(" ", "_")}_spectrum.png')
    plt.close()

# Plot frequency spectra
plot_frequency_analysis(u_signal, "Lorenz Signal")
plot_frequency_analysis(m_mono, "Monochromatic Message")
plot_frequency_analysis(m_text, "Text Message")