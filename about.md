---
layout: page
title: About
permalink: /about/
eyebrow: Background
intro: I work where systems engineering, numerical algorithms, and performance analysis meet.
description: Background, research interests, and engineering approach of Jiyu Liu.
---

<section class="content-section">
  <h2>A mathematical route into systems</h2>
  <p class="lead-copy">My doctoral research at Zhejiang University focuses on high-order numerical methods for partial differential equations on complex geometries. Building those solvers meant going beyond derivations: I designed C++ abstractions for geometry and operators, debugged sparse linear systems, parallelized kernels with OpenMP, and built convergence and performance experiments that could survive repeated refinement.</p>
  <p>That work shaped how I approach engineering problems. I first make the computational model explicit, then identify correctness boundaries, measure the real execution path, and optimize only where the data points.</p>
</section>

<section class="content-section">
  <h2>From numerical solvers to distributed execution</h2>
  <p>During my internship at Tencent's Distributed Database R&amp;D Center, I applied the same approach to a production MPP analytical engine. My work covered distributed aggregation and complex-plan adaptation, query profiling across execution layers, and compute-node reliability. The domain changed; the central questions did not: where is state owned, which work can run independently, how is correctness preserved across boundaries, and what does the profiler actually prove?</p>
</section>

<section class="content-section">
  <h2>Current direction</h2>
  <p>I am targeting AI infrastructure, performance-critical C++ systems, and quantitative-development roles. My current study extends from PyTorch and deep-learning fundamentals toward LLM inference—prefill/decode behavior, KV-cache management, prefix caching, serving schedulers, and the vLLM/SGLang codebases.</p>
  <div class="notice">Learning topics are deliberately separated from production experience throughout this site.</div>
</section>

<section class="content-section">
  <h2>How I work</h2>
  <div class="principle-grid">
    <article><strong>Correctness first</strong><span>Define semantics, invariants, failure paths, and numerical validation before tuning.</span></article>
    <article><strong>Measure by layer</strong><span>Separate setup from steady state, and global latency from operator-, node-, and thread-level behavior.</span></article>
    <article><strong>Design for reuse</strong><span>Move invariant work out of hot paths, cache expensive state, and keep fallbacks explicit.</span></article>
    <article><strong>State evidence clearly</strong><span>Report the benchmark scope and avoid turning team context or future study into personal results.</span></article>
  </div>
</section>

<section class="content-section">
  <h2>Skills in context</h2>
  <div class="skill-grid">
    {% for skill in site.data.skills %}
      <article class="skill-card">
        <h3>{{ skill.group }}</h3>
        <p>{{ skill.context }}</p>
        <ul>{% for item in skill.items %}<li>{{ item }}</li>{% endfor %}</ul>
      </article>
    {% endfor %}
  </div>
</section>
