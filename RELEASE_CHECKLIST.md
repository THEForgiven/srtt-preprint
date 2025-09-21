# Release & Zenodo Checklist

1. **Repository prep**
   - Update `README.md` (status, changelog).
   - Bump version in `CITATION.cff` and tag (e.g., v0.1-preprint).

2. **Create GitHub Release**
   - Draft a Release from the new tag.
   - Attach any large artifacts (optional).

3. **Zenodo integration**
   - Link this GitHub repo to Zenodo (https://zenodo.org/account/settings/github/).
   - Create the Release; Zenodo will mint a DOI.
   - Copy the DOI back into `CITATION.cff` and `README.md` badges.

4. **GitHub Pages**
   - Ensure Pages is set to build from `/docs` in repo settings.
   - Visit the site and verify the **Download PDF** button and math rendering.

5. **CI results**
   - Confirm the CI job `reproduce` is green.
   - Download CI artifacts to verify outputs (optional).

6. **Archive** (optional)
   - Upload the Release zip to institutional archive if required.
