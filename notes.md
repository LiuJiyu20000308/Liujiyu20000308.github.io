---
layout: page
title: Technical Notes
permalink: /notes/
intro: Notes on C++, systems, algorithms, probability, and quantitative foundations.
description: Technical study notes by Jiyu Liu.
---

<p class="archive-context">These are learning records rather than polished tutorials. Older material is kept because it documents how my technical foundation developed.</p>

<div class="notes-list">
  {% assign visible_posts = site.posts | where_exp: 'post', 'post.portfolio_hidden != true' %}
  {% for post in visible_posts %}
    <article class="note-row">
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%Y.%m.%d' }}</time>
      <div>
        <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
        {% if post.tags %}<p>{{ post.tags | join: ' · ' }}</p>{% endif %}
      </div>
    </article>
  {% endfor %}
</div>
