unique_frequencies = [
    2.54e-01, 3.40e-01, 4.56e-01, 6.12e-01, 8.22e-01, 9.99e-01, 1.10e+00, 1.33e+00,
    1.48e+00, 1.78e+00, 1.99e+00, 2.37e+00, 2.66e+00, 3.16e+00, 3.57e+00, 4.22e+00,
    4.80e+00, 5.62e+00, 6.43e+00, 7.50e+00, 8.64e+00, 1.00e+01, 1.16e+01, 1.33e+01,
    1.55e+01, 1.78e+01, 2.09e+01, 2.37e+01, 2.80e+01, 3.16e+01, 3.75e+01, 4.22e+01,
    5.03e+01, 5.62e+01, 6.76e+01, 7.50e+01, 9.06e+01, 1.02e+02, 1.22e+02, 1.35e+02,
    1.63e+02, 1.78e+02, 2.19e+02, 2.37e+02, 2.94e+02, 3.16e+02, 3.94e+02, 4.22e+02,
    5.29e+02, 5.64e+02, 7.10e+02, 7.50e+02, 9.52e+02, 1.00e+03, 1.28e+03, 1.33e+03,
    1.71e+03, 1.78e+03, 2.30e+03, 2.37e+03, 3.09e+03, 3.16e+03, 4.14e+03, 4.22e+03,
    5.56e+03, 5.62e+03, 7.45e+03, 7.50e+03, 1.00e+04
]


def get_frequencies_to_use(frequency_selection):
    match frequency_selection:
        case "physics": return physics_frequencies()
        case "feature": return feature_analysis()
        case _: return unique_frequencies  # Default fallback
    
    
def physics_frequencies(): return [f for f in unique_frequencies if 1 <= f <= 10]

def feature_analysis(): return [1.78, 2.37, 3.16, 5.62, 10.0]