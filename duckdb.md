---
layout: page
title: DuckDB 技术文档
permalink: /technical-notes/duckdb/
intro: 从 SQL 进入系统到向量化执行与性能诊断，沿查询生命周期梳理 DuckDB 的核心机制。
description: Jiyu Liu 的 DuckDB 查询引擎源码学习笔记，涵盖 Binder、优化器、向量化执行、Pipeline、Join、内存管理与 Profiling。
lang: zh-CN
---

<p class="archive-context">这组文档来自我对 DuckDB 执行引擎和相关源码的持续阅读。重点不是罗列类名，而是解释各模块为什么存在、如何协作，以及性能问题应该怎样定位。内容仅讨论 DuckDB 上游公开机制，不包含实习公司的内部实现或适配细节。</p>

<div class="duckdb-roadmap" aria-label="DuckDB reading roadmap">
  <span>SQL</span><span>Binder</span><span>Logical Plan</span><span>Optimizer</span><span>Physical Plan</span><span>Pipeline</span><span>DataChunk</span>
</div>

<div class="chapter-list">
  {% assign chapters = site.duckdb | sort: 'order' %}
  {% for chapter in chapters %}
    <article class="chapter-row">
      <span>{{ chapter.order }}</span>
      <div>
        <h2><a href="{{ chapter.url | relative_url }}">{{ chapter.title }}</a></h2>
        <p>{{ chapter.summary }}</p>
        {% if chapter.keywords %}<small>{{ chapter.keywords | join: ' · ' }}</small>{% endif %}
      </div>
    </article>
  {% endfor %}
</div>

<p class="content-note">阅读建议：先读第 1 章建立全局结构，再重点阅读第 4、5、6 章理解执行引擎。第 8 章可以作为日常性能排查清单独立使用。</p>
