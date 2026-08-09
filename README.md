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

```bash
bundle install
bundle exec jekyll serve --livereload
```

The generated `_site/` and Sass caches are intentionally ignored. GitHub Pages
builds the site from the source files on the default branch.

## Main routes

- `/` — academic-style bio, selected experience, projects, and publication;
- `/experience/` — work history and impact;
- `/projects/` — systems and numerical-computing work;
- `/notes/` — technical notes and retained learning archive;
- `/assets/Liujiyu_CV.pdf` — canonical résumé PDF.
