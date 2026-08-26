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
from providers.local_embedding_provider import LocalEmbeddingProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("PDFPipelineRunner")


async def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Local (AWS/In-Process): python scripts/run_pdf_pipeline.py <path_to_pdf>")
        print("  Remote (Colab URL):     python scripts/run_pdf_pipeline.py <path_to_pdf> --remote <remote_base_url>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pipeline = PDFChunkingPipeline()

    try:
        if len(sys.argv) >= 4 and sys.argv[2] == "--remote":
            remote_base_url = sys.argv[3]
            logger.info(f"Running Remote PDF Chunking Pipeline for '{pdf_path}' -> '{remote_base_url}'...")
            final_chunks = await pipeline.process_pdf_remote(
                pdf_path=pdf_path,
                remote_base_url=remote_base_url,
            )
        else:
            logger.info(f"Running Local in-process PDF Chunking Pipeline for '{pdf_path}'...")
            embedding_provider = LocalEmbeddingProvider()
            final_chunks = await pipeline.process_pdf_local(
                pdf_path=pdf_path,
                embedding_provider=embedding_provider,
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
