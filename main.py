"""
main.py - RAG Pipeline Orchestrator
====================================
Runs the complete pipeline: ingest → chunk → embed → index

Usage:
  python main.py
  python main.py --skip-ingest
  python main.py --skip-embeddings
  python main.py --skip-ingest --skip-embeddings
  python main.py --quiet
"""

import os
import sys
import time
from pathlib import Path

# Import pipeline modules
import ingest_sources
import chunk_text
import embed_store
import vector_store


class PipelineConfig:
    """Configuration for the RAG pipeline."""

    # Directory paths
    RAW_DIR = "data/raw"
    EMBEDDINGS_DIR = "data/embeddings"
    CHROMA_DIR = "./chroma_db"

    # File paths
    CHUNKS_PATH = "data/chunks.json"
    EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"
    METADATA_PATH = "data/embeddings/metadata.json"

    @classmethod
    def print_paths(cls):
        """Print all configured paths."""
        print("\n" + "=" * 60)
        print("PIPELINE CONFIGURATION")
        print("=" * 60)
        print(f"Raw data:        {cls.RAW_DIR}")
        print(f"Chunks:          {cls.CHUNKS_PATH}")
        print(f"Embeddings:      {cls.EMBEDDINGS_PATH}")
        print(f"Metadata:        {cls.METADATA_PATH}")
        print(f"Vector Index:    {cls.CHROMA_DIR}")
        print("=" * 60 + "\n")


class PipelineRunner:
    """Orchestrates the RAG pipeline execution."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.timings = {}

    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def run_step(self, step_name, step_function, *args, **kwargs):
        """
        Execute a pipeline step with timing and error handling.
        """
        self.log(f"\n{'=' * 60}")
        self.log(f"STEP: {step_name}")
        self.log(f"{'=' * 60}")

        start_time = time.time()

        try:
            result = step_function(*args, **kwargs)
            elapsed = time.time() - start_time
            self.timings[step_name] = elapsed

            self.log(f"✓ {step_name} completed in {elapsed:.2f}s")
            return result

        except Exception as e:
            self.log(f"✗ {step_name} FAILED: {str(e)}")
            raise

    def run_full_pipeline(self, skip_ingest=False, skip_embeddings=False):
        """
        Run the complete RAG pipeline.

        Args:
            skip_ingest: If True, skip ingestion step
            skip_embeddings: If True, skip embedding step IF embedding files exist
        """
        self.log("\n" + "🚀 " * 20)
        self.log("STARTING RAG PIPELINE")
        self.log("🚀 " * 20)

        PipelineConfig.print_paths()

        # Step 1: Ingest sources
        if not skip_ingest:
            self.run_step("1. Ingest Sources", ingest_sources.main)
        else:
            self.log("\n⏭️  Skipping ingestion step (--skip-ingest)")

        # Step 2: Chunk text
        self.run_step("2. Chunk Text", chunk_text.process_all_files)

        # Step 3: Generate embeddings (optionally skip if already present)
        embeddings_exist = (
            os.path.exists(PipelineConfig.EMBEDDINGS_PATH)
            and os.path.exists(PipelineConfig.METADATA_PATH)
        )

        if skip_embeddings and embeddings_exist:
            self.log("\n⏭️  Skipping embeddings step (--skip-embeddings) (files already exist)")
        else:
            if skip_embeddings and not embeddings_exist:
                self.log("\nℹ️  --skip-embeddings set but embeddings files not found; generating embeddings anyway.")
            self.run_step("3. Generate Embeddings", self._run_embed_store)

        # Step 4: Build vector index
        self.run_step("4. Build Vector Index", self._run_vector_store)

        # Print summary
        self._print_summary()

    def _run_embed_store(self):
        """Execute the embedding generation step."""
        chunks = embed_store.load_chunks()
        model = embed_store.load_model()
        embeddings = embed_store.create_embeddings(chunks, model)
        embed_store.save_embeddings_and_metadata(embeddings, chunks)

    def _run_vector_store(self):
        """Execute the vector indexing step."""
        texts, embeddings, ids, metadata = vector_store.load_data()
        vector_store.store_in_chroma(texts, embeddings, ids, metadata)

    def _print_summary(self):
        """Print pipeline execution summary."""
        total_time = sum(self.timings.values())

        self.log("\n" + "=" * 60)
        self.log("PIPELINE SUMMARY")
        self.log("=" * 60)

        for step, elapsed in self.timings.items():
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            self.log(f"{step:.<50} {elapsed:>6.2f}s ({percentage:>5.1f}%)")

        self.log(f"{'TOTAL TIME':.<50} {total_time:>6.2f}s")
        self.log("=" * 60)

        self.log("\n✅ Pipeline completed successfully!")
        self.log(f"📊 Vector database ready at: {PipelineConfig.CHROMA_DIR}")
        self.log("🔍 You can now run queries using rag_core.py or app.py")


def verify_dependencies():
    """Check if all required files and directories exist."""
    print("Checking pipeline dependencies...")

    issues = []

    # Check if raw sources exist
    if not os.path.exists(PipelineConfig.RAW_DIR):
        issues.append(f"❌ Raw data directory not found: {PipelineConfig.RAW_DIR}")

    # Check if required scripts exist
    required_files = [
        "ingest_sources.py",
        "chunk_text.py",
        "embed_store.py",
        "vector_store.py",
    ]

    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"❌ Required file not found: {file}")

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("✅ All dependencies found\n")
    return True


def main():
    """Main entry point for the pipeline."""

    print(
        """
    ╔══════════════════════════════════════════════════════════╗
    ║          RAG PIPELINE - QUERY BUDDY PROJECT             ║
    ║                                                          ║
    ║  Ingest → Chunk → Embed → Index                         ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )

    # Verify dependencies
    if not verify_dependencies():
        print("\n❌ Please fix the issues above before running the pipeline.")
        sys.exit(1)

    # Parse command line arguments
    skip_ingest = "--skip-ingest" in sys.argv
    skip_embeddings = "--skip-embeddings" in sys.argv
    quiet = "--quiet" in sys.argv

    if skip_ingest:
        print("ℹ️  Running with --skip-ingest flag (will skip data ingestion)")
    if skip_embeddings:
        print("ℹ️  Running with --skip-embeddings flag (will skip embedding generation if files exist)")
    if skip_ingest or skip_embeddings:
        print()

    # Run pipeline
    try:
        runner = PipelineRunner(verbose=not quiet)
        runner.run_full_pipeline(skip_ingest=skip_ingest, skip_embeddings=skip_embeddings)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
