import logging
import os


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # AWS SDK のログを抑制（冗長すぎるため）
    logging.getLogger("aws_sdk_bedrock_runtime").setLevel(logging.WARNING)
    logging.getLogger("awscrt").setLevel(logging.WARNING)
    logging.getLogger("smithy_aws_core").setLevel(logging.WARNING)
    logging.getLogger("smithy_core").setLevel(logging.WARNING)
    logging.getLogger("smithy_http").setLevel(logging.WARNING)
    logging.getLogger("smithy_aws_event_stream").setLevel(logging.WARNING)
    logging.getLogger("smithy_json").setLevel(logging.WARNING)

    # すべての AWS 関連ライブラリのログレベルを一括で上げる
    for name in logging.root.manager.loggerDict:
        if any(x in name.lower() for x in ["aws", "smithy", "boto"]):
            logging.getLogger(name).setLevel(logging.WARNING)

