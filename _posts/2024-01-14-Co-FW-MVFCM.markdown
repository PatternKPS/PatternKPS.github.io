---
layout:     post
title:      "Collaborative feature-weighted multi-view fuzzy c-means clustering"
date:       2024-03-14 11:18:00
author:     "Kristina"
header-img: "asset/Co-FL-MV-FCM.png"
tags:
    - Co-FW-MVFCM
    - Multi-view clustering
    - Feature reduction
    - Collaborative learning
    - MV-FCM
    - My publication
---

<div class="content">
<p>
Fuzzy c-means (FCM) clustering had been extended for handling multi-view data with collaborative idea. However, these collaborative multi-view FCM treats multi-view data under equal importance of feature components. In general, different features should take different weights for clustering real multi-view data. In this paper, we propose a novel multi-view FCM (MVFCM) clustering algorithm with view and feature weights based on collaborative learning, called collaborative feature-weighted MVFCM (Co-FW-MVFCM). The Co-FW-MVFCM contains a two-step schema that includes a local step and a collaborative step. The local step is a single-view partition process to produce local partition clustering in each view, and the collaborative step is sharing information of their memberships between different views. These two steps are then continuing by an aggregation way to get a global result after collaboration. Furthermore, the embedded feature-weighted procedure in Co-FW-MVFCM can give feature reduction to exclude redundant/irrelevant feature components during clustering processes. Experiments with several data sets demonstrate that the proposed Co-FW-MVFCM algorithm can completely identify irrelevant feature components in each view and that, additionally, it can improve the performance of the algorithm. Comparisons of Co-FW-MVFCM with some existing MVFCM algorithms are made and also demonstrated the effectiveness and usefulness of the proposed Co-FW-MVFCM clustering algorithm.</p>
<ul class="actions">
<li><a href="https://www.sciencedirect.com/science/article/abs/pii/S003132032100251X" class="button"
style="color: black;background-color: rgba(75, 75, 76, 0.100);">Paper</a></li>
<li><a href="https://github.com/kpnaga08/Co-FW-MVFCM" class="button"
style="color: black;background-color: rgba(75, 75, 76, 0.100);">Codes</a>
</li>
</ul>
</div>
