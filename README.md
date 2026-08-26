# Jiyu Liu — Portfolio

Personal portfolio and technical-notes archive, built with Jekyll 3.9 and
deployed through GitHub Pages.

## Content maintenance

The portfolio is intentionally data-driven:

- `_data/profile.yml`: homepage identity, focus areas, and headline metrics;
- `_data/experience.yml`: chronological work experience;
- `_data/research.yml`: research cases, contributions, and verified results;
- `_data/projects.yml`: selected engineering case studies;
- `_data/skills.yml`: skills grouped by real usage context;
- `_data/education.yml` and `_data/honors.yml`: résumé data;
- `_posts/`: historical technical notes.

`assets/Liujiyu_CV.pdf` is the canonical résumé. It is copied directly from
the author-maintained PDF rather than generated from an HTML résumé page.

Update these files instead of duplicating content across pages. Do not publish
private repository links or unconfirmed publication metadata.

## Local build

The site uses a dedicated Conda environment so that Ruby and the native build
tools required by several gems do not depend on the system installation. Create
it once with:

```bash
conda create --override-channels -c conda-forge -n jekyll \
  ruby=3.1 gcc_linux-64 gxx_linux-64 make pkg-config openssl libffi
```

Then build or preview the site with:

```bash
conda activate jekyll
bundle install
bundle exec jekyll build
bundle exec jekyll serve --livereload
```

The preview is available at `http://127.0.0.1:4000/`. After the first
installation, later sessions only need `conda activate jekyll` before running
the `bundle exec` commands.

The generated `_site/` and Sass caches are intentionally ignored. GitHub Pages
builds the site from the source files on the default branch.

## Main routes

- `/` — academic-style bio, selected experience, projects, and publication;
- `/experience/` — work history and impact;
- `/projects/` — systems and numerical-computing work;
- `/notes/` — technical notes and retained learning archive;
- `/assets/Liujiyu_CV.pdf` — canonical résumé PDF.
