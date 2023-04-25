def detect_degradation(values, sample_size=10, degradation_margin=0.1):
    if len(values) <= sample_size:
        return False

    sample = values[-sample_size:]
    average = sum(sample) / len(sample)
    return average < (1 - degradation_margin) * max(sample)
