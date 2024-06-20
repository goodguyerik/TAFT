def send_metrics(ex, metric_to_value: dict):
    for metric, value in metric_to_value.items():
        if isinstance(value, list):
            for idx, step_value in enumerate(value):
                ex.log_scalar(metric, step_value, idx)
        else:
            ex.log_scalar(metric, value)
