"""Rebuild everything behind the paper.

    python run_all.py              simulations, figures, numbers, Word manuscript
    python run_all.py --real-data  also re-run the LFW and CelebA-Spoof benchmarks
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd):
    print(f"\n> {' '.join(cmd)}   (in {os.path.relpath(cwd, ROOT)})")
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"Failed: {' '.join(cmd)}")


def main():
    exp = os.path.join(ROOT, "research", "experiments")
    word = os.path.join(ROOT, "manuscript", "word_build")
    run([sys.executable, "run_all.py"] + (["--real-data"] if "--real-data" in sys.argv else []), exp)
    supp = os.path.join(ROOT, "manuscript", "supplement")
    if os.path.isdir(supp):
        run([sys.executable, "make_supplement.py"], supp)   # then: pdflatex supplement.tex (twice)
    if not os.path.isdir(word):
        print("\nResults, figures and manuscript/generated/ are up to date (no manuscript sources here, so no Word build).")
        return
    run([sys.executable, "prep_content.py"], word)
    if shutil.which("node"):
        env_double = dict(os.environ, LAYOUT="double")
        print("\n> node build_manuscript.js Manuscript_TwoColumn.docx   (IEEE two-column)")
        if subprocess.run(["node", "build_manuscript.js", "Manuscript_TwoColumn.docx"], cwd=word, env=env_double).returncode != 0:
            sys.exit("Failed: two-column build")
        run(["node", "build_manuscript.js", "Manuscript_SingleColumn.docx"], word)
        run(["node", "build_cover_letter.js", "Cover_Letter.docx"], word)
        print("\nWord files written to manuscript/word_build/. Copy them into submission/ when final.")
    else:
        print("\nNode.js not found: skipped the Word build (install Node.js and run `npm install docx`).")


if __name__ == "__main__":
    main()
