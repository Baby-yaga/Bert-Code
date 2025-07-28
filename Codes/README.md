# 基于微调参数的Bert模型进行文本分类的示例代码说明
所谓微调参数，是指在基础的Bert模型基础上，运用自主构建的训练集帮助Bert模型更加精准的识别文本含义的过程。文本分类是指，对需要预测（分类）的文本批量的打标签操作，比较常见的包含文本情绪识别，如积极或消极；是否涉及相关含义，如是否包含ESG岗位相关职能，是否包含绿色产品，是否包含数字化发展等内容。整个流程如下：
## 首先是构造预训练集。
构造预训练集是微调模型参数的必要前提，可以采用人工或生成式人工智能结合的方式进行构造。  

生成式人工智能构造，首推这种方法。可以参考环境配置文档中给出的实例（还有Step1和Step2中的预训练集构造部分），根据任务目标直接对本地大模型进行批量提问，进而生成预训练集。接着我们对预训练集进行必要的抽查。

人工构造，顾名思义即根据任务目标手动构造标签，但这显然费事费力。具体可以参考张大永、姬强等发表在Nature旗下期刊的气候政策不确定性的构造思路（相关内容如下，供参考）：  
  
CCPU refers to the uncertainty associated with various aspects of climate policies, including the entities responsible for policy formulation (who develops climate policies), the issuance timing and policy content (when and what types of policies are implemented) and the consequences of the implementation of climate policies (outcomes of climate policy actions). A manual auditing team comprising master’s and doctoral students in the field of economics and finance is first assembled to determine whether a news item contains uncertainty on climate policy. Each member is asked to assess whether a news item contained a CCPU by manually reading the whole context of the article. The news item was then labelled CPU = 1 if yes, indicating that the news item contained CPU. Each item is assessed independently by multiple readers to ensure reliability. The detailed steps are as follows:  

CCPU 指的是与气候政策各个方面相关的不确定性，包括负责政策制定的实体（由谁制定气候政策）、发布时间和政策内容（何时以及实施何种类型的政策）以及气候政策实施的后果（气候政策行动的结果）。首先组建一个由经济学和金融学领域的硕士生和博士生组成的人工审核小组，以确定一条新闻是否包含气候政策的不确定性。要求每位成员通过手动阅读文章的整个上下文来评估新闻条目是否包含 CCPU。如果是，则标注 CPU = 1，表示该新闻条目包含 CPU。每个条目都由多名阅读者独立评估，以确保可靠性。具体步骤如下：  

Form an auditing guide. Each auditing team member reads news related to climate policy and invests two months in this endeavour. During this process, team members collectively formulate a standardised approach to recording the results of the manual audits. A manual assessment guide is also developed to ensure consistency and accuracy in the assessment process. The guideline provides detailed explanations of the auditing rules, an assessment template, frequently asked questions and several examples of auditing studies. These resources are provided to help the auditing team better understand the requirements for completing manual auditing accurately.  

形成审计指南。每位审计小组成员都会阅读与气候政策相关的新闻，并投入两个月的时间进行这项工作。在此过程中，团队成员共同制定了记录人工审核结果的标准化方法。还制定了人工评估指南，以确保评估过程的一致性和准确性。该指南详细解释了审核规则、评估模板、常见问题和一些审核研究实例。提供这些资源是为了帮助审核团队更好地理解准确完成人工审核的要求。  

Training and pre-assessment. Next, an auditing team consisting of 48 master’s and doctoral students specialising in economics or finance from universities such as the University of Chinese Academy of Sciences, Southwestern University of Finance and Economics, China University of Mining and Technology and the University of Science and Technology of Macau is created. Relevant training is provided to the auditing team members, and each is assigned 100 news articles to rate. Based on the preliminary assessment results, further training is provided to the team members, and the guidelines undergo continued revision and refinement. Following several iterations, a pre-assessment result accuracy rate higher than 96% is achieved.  

培训和预评估。其次，从中国科学院大学、西南财经大学、中国矿业大学、澳门科技大学等高校选拔 48 名经济学或金融学专业的硕士生和博士生组成审计团队。对审核组成员进行相关培训，每人分配 100 篇新闻进行评分。根据初步评估结果，对小组成员进行进一步培训，并对指南进行持续修订和完善。经过多次反复，预评估结果的准确率超过了 96%。  

Formal auditing. Each team member is assigned 800 news items to evaluate. The assessment results are then used for subsequent training and evaluation of the deep learning models. To increase the efficiency of the formal process, news items that have already been read are removed, and then a total of 28,800 news items are randomly and proportionally selected as the reading sample. Forty-eight auditors are assigned to 16 teams to make sure that each news item would be read by three auditors independently. During the formal manual auditing phase, group discussions are scheduled to recapitulate the challenges faced during the auditing process and continually improve the guidelines. A total of 4 months are required for all auditors to finish reading and rating the news samples.  

正式审核。每个团队成员分配到 800 个新闻项目进行评估。评估结果将用于深度学习模型的后续训练和评估。为了提高正式流程的效率，先剔除已经阅读过的新闻条目，然后按比例随机抽取总共 28800 条新闻条目作为阅读样本。48 名审核员被分配到 16 个小组，以确保每个新闻条目由 3 名审核员独立阅读。在正式的人工审核阶段，会安排小组讨论，总结审核过程中面临的挑战，不断改进指南。所有审核员总共需要 4 个月的时间来完成新闻样本的阅读和评级。  

Cross-validation. News items with inconsistent results among the three auditors in the auditing teams are reassigned to another group of auditors for additional assessment. The new results are returned to the original team for feedback and to ensure consistency. In addition, the research team spent 1 month examining the manual auditing results.  

最后进行交叉验证  
  
## 接着是导入数据，具体可参考data_process.py
## 再次是模型训练部分，具体可参考train_and_eval.py
## 最后是模型预测部分，具体可参考modeling.py，这也是整个模型训练的核心部分。
