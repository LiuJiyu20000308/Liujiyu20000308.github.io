---
layout: page
title: 中文简历
permalink: /resume-zh/
intro: 面向公开网站的脱敏版本，不包含电话号码、内部项目代号或未公开实现细节。
description: 刘骥宇的中文公开简历，涵盖分布式数据库研发与计算数学研究。
body_class: resume-page
lang: zh-CN
---

<div class="resume-actions" data-nosnippet>
  <a class="button button-primary" href="{{ site.data.profile.resume_pdf_zh | relative_url }}" download>下载中文 PDF</a>
  <a class="button button-secondary" href="{{ site.data.profile.resume_pdf_en | relative_url }}" download>English PDF</a>
  <button class="button button-secondary" type="button" onclick="window.print()">打印本页</button>
</div>

<article class="resume-document resume-document-zh">
  <header class="resume-header">
    <div>
      <h2>刘骥宇 <span>Jiyu Liu</span></h2>
      <p>分布式数据库研发 · 高性能 C++ · 计算数学</p>
    </div>
    <div class="resume-contact">
      <a href="mailto:{{ site.data.profile.email }}">{{ site.data.profile.email }}</a>
      <a href="{{ site.data.profile.github }}">github.com/LiuJiyu20000308</a>
    </div>
  </header>

  <section>
    <h3>教育经历</h3>
    <div class="resume-entry">
      <div class="resume-entry-heading"><h4>浙江大学 · 计算数学 · 全日制直博在读</h4><span>2021.09–预计 2027.03</span></div>
      <p>GPA 92/100；博士学位尚未授予；研究方向为高性能数值计算、有限体积方法与多重网格。</p>
    </div>
    <div class="resume-entry">
      <div class="resume-entry-heading"><h4>浙江大学 · 信息与计算科学 · 理学学士</h4><span>2017.09–2021.06</span></div>
      <p>总 GPA 4.38/5.00，后两年 4.78/5.00；浙江大学优秀毕业生（Top 5%）、一等奖学金（Top 2%）。</p>
    </div>
  </section>

  <section>
    <h3>实习经历</h3>
    <div class="resume-entry">
      <div class="resume-entry-heading"><h4>腾讯 · 分布式数据库研发实习生（青云计划）</h4><span>2025.07–2026.07</span></div>
      <ul>
        <li>参与生产级 MPP 分析数据库研发，完成两阶段聚合与复杂子查询的分布式计划适配，处理聚合中间状态、数据重分布和跨节点语义一致性问题。</li>
        <li>开发分布式性能分析链路，整合 query/operator/node/thread 四级指标；通过大规模分析型负载定位高频采集引入的锁竞争，并调整采集边界控制诊断开销。</li>
        <li>参与计算节点生命周期与故障恢复建设，覆盖注册、心跳、版本化状态收敛、分布式锁、查询重试和集成测试。</li>
      </ul>
    </div>
    <div class="resume-entry">
      <div class="resume-entry-heading"><h4>杭州希格斯投资管理有限公司 · 全栈策略工程师</h4><span>2024.11–2025.06</span></div>
      <ul>
        <li>基于 C++/Python、SQLite 与 TDengine 建设行情数据存储和实时订阅链路，支持多线程、多进程并发访问，并对接券商行情 API。</li>
        <li>开发交易监控、数据查询和本地知识库工具，用于量化研究与日常运维问题分析。</li>
      </ul>
    </div>
  </section>

  <section>
    <h3>科研与项目</h3>
    <div class="resume-entry">
      <div class="resume-entry-heading"><h4>四阶 Cut-cell 多重网格椭圆方程求解器</h4><span>C++ · OpenMP</span></div>
      <ul>
        <li>基于 Cut-cell、PLG 多项式重构与加权最小二乘构造复杂几何上的四阶有限体积离散，并设计几何多重网格与不规则单元 block relaxation。</li>
        <li>在多类复杂几何上获得约四阶收敛和近似最优网格加密复杂度；引入稀疏底层求解后，将一个基准算例从 118.770 s 降至 4.441 s（26.7×）。</li>
        <li><a href="https://arxiv.org/abs/2601.02975">arXiv:2601.02975</a>，论文尚未投稿期刊。</li>
      </ul>
    </div>
    <div class="resume-entry">
      <div class="resume-entry-heading"><h4>密度分层流体高阶求解器</h4><span>C++ · OpenMP · MKL</span></div>
      <ul>
        <li>实现二维复杂几何上的不可压 Navier–Stokes 高阶有限体积求解链路，结合压力投影、几何多重网格和并行数值计算。</li>
        <li>在制造解实验中验证时空四阶精度，数值结果与参考实验保持一致。</li>
      </ul>
    </div>
  </section>

  <section>
    <h3>技术能力</h3>
    <div class="resume-skills">
      <p><strong>系统研发：</strong>C++、Python、SQL；Linux、Git、CMake、GDB；并发编程、性能分析与回归测试。</p>
      <p><strong>数据库与基础设施：</strong>SQL 执行引擎与优化器、MPP 执行、数据交换、流水线调度、SQLite、TDengine。</p>
      <p><strong>科学计算：</strong>有限体积、几何多重网格、稀疏线性代数、OpenMP、Eigen、LAPACK/MKL、HDF5/Silo。</p>
    </div>
  </section>

  <section>
    <h3>荣誉</h3>
    <p>浙江大学优秀毕业生（Top 5%） · 浙江大学一等奖学金（Top 2%） · CASC 二等奖学金 · 优秀研究生 · 优秀研究生干部 · 国祥奖学金</p>
  </section>
</article>
