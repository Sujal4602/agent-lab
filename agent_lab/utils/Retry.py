from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def groq_retry(fn):
    return retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )(fn)