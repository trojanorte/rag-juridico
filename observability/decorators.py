from observability.telemetry import telemetry


def measure(metric_name):

    def decorator(func):

        def wrapper(*args, **kwargs):

            start = telemetry.start_timer()

            try:
                result = func(*args, **kwargs)

                duration = telemetry.stop_timer(start)
                telemetry.metrics[metric_name] = duration

                return result

            except Exception as e:
                telemetry.metrics["error"] = str(e)
                raise e

        return wrapper

    return decorator