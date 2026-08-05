---
layout: page
title: Projects
permalink: /projects/
intro: Selected systems and numerical-computing work, described by the problem, my contribution, and the result.
description: Selected distributed systems, HPC, and quantitative infrastructure projects by Jiyu Liu.
---

<div class="project-list">
  {% for project in site.data.projects %}
    <article class="project-entry" id="{{ project.title | slugify }}">
      <header>
        <p class="entry-label">{{ project.category }}</p>
        <h2>{{ project.title }}</h2>
      </header>
      <div class="project-details">
        <section>
          <h3>Problem</h3>
          <p>{{ project.problem }}</p>
        </section>
        <section>
          <h3>Work</h3>
          <p>{{ project.approach }}</p>
        </section>
        <section>
          <h3>Result</h3>
          <p>{{ project.outcome }}</p>
        </section>
      </div>
      <ul class="tag-list">
        {% for technology in project.technologies %}<li>{{ technology }}</li>{% endfor %}
      </ul>
    </article>
  {% endfor %}
</div>

<p class="content-note">Company source code and unpublished research repositories are not linked publicly.</p>
