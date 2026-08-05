---
layout: page
title: Résumé
permalink: /resume/
eyebrow: Profile
intro: A concise, public version of my experience, with a privacy-safe PDF for direct download.
description: Public résumé of Jiyu Liu, covering distributed database engineering and computational mathematics.
body_class: resume-page
---

<div class="resume-actions" data-nosnippet>
  <a class="button button-primary" href="{{ site.data.profile.resume_pdf_en | relative_url }}" download>Download English PDF</a>
  <a class="button button-secondary" href="{{ site.data.profile.resume_pdf_zh | relative_url }}" download>下载中文 PDF</a>
  <button class="button button-secondary" type="button" onclick="window.print()">Print this page</button>
  <a class="button button-secondary" href="mailto:{{ site.data.profile.email }}">Request a tailored résumé</a>
</div>

<article class="resume-document">
  <header class="resume-header">
    <div>
      <h2>Jiyu Liu <span lang="zh-CN">刘骥宇</span></h2>
      <p>Distributed Database Engineering · High-Performance C++ · Computational Mathematics</p>
    </div>
    <div class="resume-contact">
      <a href="mailto:{{ site.data.profile.email }}">{{ site.data.profile.email }}</a>
      <a href="{{ site.data.profile.github }}">github.com/LiuJiyu20000308</a>
    </div>
  </header>

  <section>
    <h3>Profile</h3>
    <p>Systems engineer with doctoral research experience in computational mathematics. Built distributed query-execution and profiling capabilities in a production MPP engine, alongside fourth-order C++ finite-volume and multigrid solvers for complex geometries.</p>
  </section>

  <section>
    <h3>Experience</h3>
    {% for job in site.data.experience %}
      <div class="resume-entry">
        <div class="resume-entry-heading">
          <h4>{{ job.company }} · {{ job.role }}</h4>
          <span>{{ job.dates }}</span>
        </div>
        <ul>{% for bullet in job.bullets limit:3 %}<li>{{ bullet }}</li>{% endfor %}</ul>
      </div>
    {% endfor %}
  </section>

  <section>
    <h3>Selected Research</h3>
    {% for research in site.data.research %}
      <div class="resume-entry">
        <div class="resume-entry-heading"><h4>{{ research.title }}</h4><span>{{ research.role }}</span></div>
        <p>{{ research.summary }}</p>
        <ul>{% for result in research.results limit:2 %}<li>{{ result }}</li>{% endfor %}</ul>
        {% if research.publication %}<p><a href="{{ research.publication.url }}">{{ research.publication.venue }} · {{ research.publication.title }}</a></p>{% endif %}
      </div>
    {% endfor %}
  </section>

  <section>
    <h3>Education</h3>
    {% for item in site.data.education %}
      <div class="resume-entry">
        <div class="resume-entry-heading"><h4>{{ item.institution }} · {{ item.program }}</h4><span>{{ item.dates }}</span></div>
        <p>{{ item.details | join: ' · ' }}</p>
      </div>
    {% endfor %}
  </section>

  <section>
    <h3>Technical Scope</h3>
    <div class="resume-skills">
      {% for skill in site.data.skills limit:3 %}
        <p><strong>{{ skill.group }}:</strong> {{ skill.items | join: '; ' }}</p>
      {% endfor %}
    </div>
  </section>

  <section>
    <h3>Honors</h3>
    <p>{{ site.data.honors | join: ' · ' }}</p>
  </section>
</article>
