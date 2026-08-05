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

`assets/Jiyu_Liu_Resume.pdf` is the privacy-safe public résumé generated from
the `/resume/` page. The private application résumé is intentionally kept
outside this repository.

Update these files instead of duplicating content across pages. Do not publish
phone numbers, private repository links, internal identifiers, unpublished
implementation details, or unconfirmed publication metadata.

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
- `/resume/` — printable public résumé;
- `/assets/Jiyu_Liu_Resume.pdf` — privacy-safe downloadable CV.
