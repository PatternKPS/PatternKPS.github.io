---
layout:     post
title:      "多視圖數據模糊聚類演算法 (Multi-view fuzzy clustering algorithms for multi-view data)"
date:       2024-03-14 11:57:00
author:     "Kristina"
header-img: ""
tags:
    - Dissertation
---

<div class="content">
<p>
模糊c-均值(Fuzzy c-means, FCM)演算法已廣泛地應用在(單視圖)資料集分類上。由於社交媒體和物聯網(IoT)在實際生活上被廣泛使用，多視圖資料變得更普遍。舉例來說，可從不同的新聞媒體來源獲知相同的事件新聞。網頁能同時依據內容和引導超連結的錨點文字，一個影像能由不同的特質和特徵空間所代表。此外，多種來源的資料準備過程需要額外的工作把它們變為有條理的狀態。因此，處理多視圖資料變成一個重要的主題。關於處理多視圖資料，FCM已經透過合作理念(collaborative idea)得到擴展。然而，這些合作式多視圖FCM (multi-view FCM, MVFCM)多數以特徵成分具相同重要的情況下處理多視圖資料，在實際的多視圖資料裡，不同的特徵成分通常應該會有不同的權重。關於這些觀點，我們首先提出一個特徵加權多視圖模糊c-均值(feature-weighted MVFCM, FW-MVFCM) 聚類演算法，藉由將特徵權重帶入MVFCM來區分有意義和無意義的特徵。此外，我們在FW-MVFCM裡採用合作式學習，稱作合作式特徵加權MVFCM (Co-FW-MVFCM) 聚類演算法。Co-FW-MVFCM包含兩個步驟流程，其中包括局部步驟和合作步驟。局部步驟是一個單視圖分割過程在一個視圖裡產生局部分割，而合作步驟是在一個視圖裡的每一個局部分割或成員將分享他們的資訊給於其他的視圖。然後以聚合的方式繼續執行這兩個步驟，在合作後得到整體的結果。通常，在聚類過程中處理無意義的特徵往往無法保留資料精確結構，結局是聚類結果的效能可能下降，因此，特徵縮減是必須且要考慮的。Co-FW-MVFCM中的特徵縮減步驟是探索一個視圖的特徵成份是否有用，假若是無意義或無訊息，則在聚類過程中將被去除。我們使用好幾個資料集來實驗展示所提出的Co-FW-MVFCM演算法能完全識別一個視圖和其他視圖的不相關特徵，此外，它移除了其他特徵而沒有顯著的傷害演算法的效能。在Co-FW-MVFCM和一些已存在的MVFCM演算法比較下，也展現了所提的Co-FW-MVFCM聚類演算法的效力和用處。</p>

<div class="content">
<p>
Fuzzy c-means (FCM) algorithm had been widely applied for clustering (single-view) data sets. Since social media and Internet of Things (IoT) are widely used in real-life, multi-view data are more common. For example, the same news can be told from different news sources. Web pages can be grouped based on both content and anchor text leading to hyperlinks, and one image can be represented with different properties and different feature spaces. Moreover, the process of preparing multiple sources data requires the additional work to get them into a structured state. Therefore, handling multi-view data becomes an important topic. For handling multi-view data, FCM had been extended with a collaborative idea. However, most of these collaborative multi-view FCM (MVFCM) treats multi-view data points under equal importance of feature components. In real multi-view data, different feature components should generally take different weights for different feature components. Concerning these aspects, we first propose a feature-weighted MVFCM (FW-MVFCM) clustering algorithm to distinguish between relevant and irrelevant features by embedding feature weights into MVFCM. In addition, we adopt collaborative learning into FW-MVFCM, called a collaborative feature-weighted MVFCM (Co-FW-MVFCM) clustering algorithm. The Co-FW-MVFCM contains a two-step schema that includes a local step and a collaborative step. The local step is a single-view partition process to produce local partition in one view, and the collaborative step is that each local partition/membership in one view will share their information with another view. These two steps are then continuing by an aggregation way to get a global result after collaboration. Often, processing irrelevant features during clustering processes tends to fail in retaining the precise structure of data. As a consequence, the performance of clustering results may degrade. The feature reduction is then necessary and taken into account. A feature reduction step in Co-FW-MVFCM is exploring how useful feature components in one view if the irrelevant/un-informative features are excluding during clustering processes. Experiments with several data sets demonstrate that the proposed Co-FW-MVFCM algorithm can identify completely unrelated features in one view and that, additionally, it removes other features without significantly hurting the performance of the algorithm. Comparisons of Co-FW-MVFCM with some existing MVFCM algorithms are made and also demonstrate the effectiveness and usefulness of the proposed Co-FW-MVFCM clustering algorithm.</p>


<ul class="actions">
<li><a href="https://www.airitilibrary.com/Article/Detail/U0017-0607202010131400" class="button"
style="color: black;background-color: rgba(75, 75, 76, 0.100);">Dissertation Link</a></li>
<li><a href="https://github.com/kpnaga08/dissertation-defense-presentation-latex/blob/main/Dissertation_Presentation.pdf" class="button"
style="color: black;background-color: rgba(75, 75, 76, 0.100);">PPT</a>
</li>
</ul>
</div>

