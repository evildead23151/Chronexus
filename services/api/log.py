"""
Sentinel — Logging wrapper (structlog with stdlib fallback)
"""
import logging

try:
    import structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    def get_logger(name: str = "sentinel"):
        return structlog.get_logger(name)

except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}',
        datefmt='%Y-%m-%dT%H:%M:%S',
    )

    def get_logger(name: str = "sentinel"):
        return logging.getLogger(name)
