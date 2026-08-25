import sys
import asyncio
import json
import logging
from pathlib import Path

# Add src directory to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from services.pdf_chunking_pipeline import PDFChunkingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("PDFPipelineRunner")


async def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_pdf_pipeline.py <path_to_pdf> <remote_base_url>")
        print("Example: python scripts/run_pdf_pipeline.py src/src/assets/sample.pdf https://xyz.ngrok-free.app")
        sys.exit(1)

    pdf_path = sys.argv[1]
    remote_base_url = sys.argv[2]

    logger.info(f"Initializing Remote-PDF Chunking Pipeline for '{pdf_path}'...")
    pipeline = PDFChunkingPipeline()

    try:
        final_chunks = await pipeline.process_pdf_remote(
            pdf_path=pdf_path,
            remote_base_url=remote_base_url,
        )
        logger.info(f"Pipeline complete! Outputted {len(final_chunks)} ready-to-index DocumentChunk objects.")

        if final_chunks:
            print("\n=================== SAMPLE FINAL CHUNK OBJECT (JSON) ===================")
            print(json.dumps(final_chunks[0].model_dump(), indent=2))
            print("========================================================================")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
