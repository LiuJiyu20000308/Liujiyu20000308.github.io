---
layout: page
title: Experience
permalink: /experience/
intro: Production engineering across distributed databases, performance diagnostics, and quantitative data infrastructure.
description: Professional experience of Jiyu Liu in distributed databases and quantitative infrastructure.
---

<div class="experience-list">
  {% for job in site.data.experience %}
    <article class="experience-entry">
      <header class="entry-header">
        <div>
          <h2>{{ job.company }}</h2>
          <p class="entry-role">{{ job.role }}</p>
          {% if job.team %}<p class="entry-team">{{ job.team }}</p>{% endif %}
        </div>
        <div class="entry-meta">
          <span>{{ job.dates }}</span>
          <span>{{ job.location }}</span>
        </div>
      </header>
      <p class="entry-summary">{{ job.summary }}</p>
      <ul class="impact-list">
        {% for bullet in job.bullets %}<li>{{ bullet }}</li>{% endfor %}
      </ul>
      <ul class="tag-list">
        {% for technology in job.technologies %}<li>{{ technology }}</li>{% endfor %}
      </ul>
    </article>
  {% endfor %}
</div>
