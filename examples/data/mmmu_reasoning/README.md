---
license: Apache License 2.0

configs:
- config_name: default
  data_files:
  - split: validation
    path: "validation.jsonl"

---

[English Version](https://modelscope.cn/datasets/modelscope/MMMU-Reasoning-Distill-Validation/file/view/master?id=80394&status=1&fileName=README_en.md)


## 描述
**MMMU-Reasoning-Distill-Validation**是一个**多模态Reasoning**数据集，包含839条图像描述和自然语言推理数据样本。该数据集基于[MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU)的验证集构建。构建流程里，首先使用Qwen2.5-VL-72B-Instruct进行图像理解并生成详细的图像描述，然后通过DeepSeek-R1模型生成自然语言推理会话。其主要特点如下：

* 使用`Perception-Reasoning` 流水线，即使用VL模型理解图像上下文语义场景并生成描述，Reasoning模型生成自然语言推理会话。
* 难度更高、面向任务更复杂，适用于"Reasoning-Intensive"的任务。
* image字段聚合，将单图推理和多图推理聚合为`images`字段，结合`messages`字段，可支持通过[ms-swift](https://github.com/modelscope/ms-swift)进行多模态GRPO训练和微调。


> 注意: 我们推荐使用该数据集来探索VL模型的Reasoning能力，而非用于刷benchmark； 同时，你可以进一步将该数据集做更细粒度的拆分，变为train/validation set，来测试模型的多模态reasoning能力。


## 样本示例

### 1. 单图推理

<div style="text-align:center;">
  <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/mmmu_r1/validation_Computer_Science_9.png" alt="image" width="600">
</div>

- Id: 
```text
validation_Computer_Science_9
```

- Question:
```text
Consider the scenario shown below, with four different servers connected to four different clients over four three-hop paths. The four pairs share a common middle hop with a transmission capacity of R = 300 Mbps. The four links from the servers to the shared link have a transmission capacity of RS = 80 Mbps. Each of the four links from the shared middle link to a client has a transmission capacity of RC = 60 Mbps. <image 1> What is the maximum achievable end-end throughput (in Mbps) for each of four client-to-server pairs, assuming that the middle link is fairly shared (divides its transmission rate equally)? Omit the unit of Mbps.
```


- Answer:
```text
60
```

- Messages:
```json
[
    {
        "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
        "role": "system"
    },
    {
        "content": "Refer to following question, description of image and given options: \n\nThe question:\nConsider the scenario shown below, with four different servers connected to four different clients over four three-hop paths. The four pairs share a common middle hop with a transmission capacity of R = 300 Mbps. The four links from the servers to the shared link have a transmission capacity of RS = 80 Mbps. Each of the four links from the shared middle link to a client has a transmission capacity of RC = 60 Mbps. <image 1> What is the maximum achievable end-end throughput (in Mbps) for each of four client-to-server pairs, assuming that the middle link is fairly shared (divides its transmission rate equally)? Omit the unit of Mbps.\n\nThe description of the image_1:\nThe image depicts a network topology diagram illustrating the connection between various devices and their data transfer rates. The central component is a cylindrical hub or switch, highlighted with a light blue glow, indicating it as the main point of connectivity. This hub has a capacity labeled as \"R = 300 Mbps,\" suggesting its maximum throughput.\n\n### Connections to Servers:\n- **Four servers** are connected to the hub via red cables. Each server is represented by a rectangular box with multiple horizontal lines inside, symbolizing hard drives or storage units. \n- The servers are positioned at the top left, top right, bottom left, and bottom right corners of the image.\n- Each server connection is labeled with \"R_S = 80 Mbps,\" indicating the data rate for each server-to-hub link.\n\n### Connections to Clients:\n- **Three client computers** (laptops) are also connected to the hub via red cables. These clients are located at the bottom left, bottom center, and bottom right of the image.\n- Each client connection is labeled with \"R_C = 60 Mbps,\" representing the data rate for each client-to-hub link.\n\n### Cables and Data Flow:\n- The red cables connecting the devices to the hub visually represent the flow of data. They are depicted as transparent tubes with red highlights, emphasizing the active transmission paths.\n- The arrows on the cables indicate the direction of data flow from the servers/clients toward the hub and vice versa.\n\n### Overall Relationship:\n- The diagram illustrates a star topology where all devices (servers and clients) are directly connected to the central hub. The hub acts as the intermediary for all data communication between the connected devices.\n- The varying data rates (\"R_S\" and \"R_C\") suggest that servers have higher bandwidth allocation compared to clients, likely due to their role in serving data-intensive tasks.\n\n The options:\n\n\n\nPlease reason step by step, and put your final answer (best option) within \\boxed{}.",
        "role": "user"
    },
    {
        "content": "<think>Okay, so I need to figure out the maximum achievable end-to-end throughput for each of the four client-to-server pairs in this network setup. Let me start by understanding the problem step by step.\n\nFirst, the network topology has four servers and four clients connected through a central hub. Each connection from the servers to the hub (RS) is 80 Mbps, and each connection from the hub to the clients (RC) is 60 Mbps. The middle link, which is the hub itself, has a capacity of R = 300 Mbps. The question states that the middle link is fairly shared, meaning it divides its transmission rate equally among the four pairs. I need to calculate the maximum throughput each pair can achieve.\n\nHmm, okay. So each client-server pair has a three-hop path: server to hub (RS), hub to client (RC), and the middle link (R) which is shared. The bottleneck could be any of these links. Since the middle link is shared and divided equally, each pair would get R/4. Let me check the numbers.\n\nFirst, let's compute the capacity allocated from the middle link for each pair. The middle link is 300 Mbps, divided by 4, so each pair gets 300/4 = 75 Mbps. Then, the server link for each pair is 80 Mbps, and the client link is 60 Mbps. So, each path has three segments: 80 Mbps, 75 Mbps (from the middle link's fair share), and 60 Mbps.\n\nBut wait, throughput is determined by the smallest capacity link in the path, right? Because the maximum data rate you can push through a path is limited by the slowest link. So in this case, the end-to-end throughput for each pair would be the minimum of RS, R/4, and RC.\n\nSo, RS is 80, R/4 is 75, and RC is 60. The minimum of these is 60 Mbps. Wait, but hold on. Let me think again. Maybe I need to consider that the middle link is being shared, but the server and client links are dedicated to each pair?\n\nWait, each server has its own link to the hub (80 Mbps), and each client has its own link from the hub (60 Mbps). So, each client-server pair has a path that goes through their server's 80 Mbps link, then through the middle hub which is shared (so each gets 75 Mbps), then through the client's 60 Mbps link. Therefore, the throughput can't exceed any of these three values. Since 60 is the smallest, that would be the maximum throughput. But maybe I'm missing something here.\n\nAlternatively, perhaps the server link and client link are directional. If data is going from server to client, then the server's upload capacity is 80 Mbps, the hub's middle link allocates 75 Mbps per pair, and the client's download capacity is 60 Mbps. So, the client's download speed would be the limiting factor here. Similarly, if it's client to server, maybe the direction is reversed? But the problem says \"end-end throughput for each of four client-to-server pairs.\" Wait, client-to-server pairs. So data is going from client to server? Wait, no, usually client-server pairs would mean the server is sending to the client, but the problem statement mentions \"client-to-server pairs,\" which is a bit confusing. Wait, the question says: \"the four different servers connected to four different clients over four three-hop paths.\" So each pair is a server and a client connected via a three-hop path. So, the direction could be either way, but throughput is considered for each pair. The problem is asking for end-end throughput, so whether it's client to server or server to client, the maximum throughput would depend on the bottleneck in that path.\n\nBut let's check the problem statement again: \"What is the maximum achievable end-end throughput (in Mbps) for each of four client-to-server pairs...\" So the pairs are client-to-server, which might imply that the direction is from client to server. Hmm, but in that case, the client's upload capacity would be RC, which is 60 Mbps, and the server's download capacity would be RS, which is 80 Mbps. But the middle link is 300 Mbps divided equally among the four pairs. So, if the direction is from client to server, then the path is client (upload 60 Mbps) -> middle link (75 Mbps) -> server (download 80 Mbps). So the bottleneck would be the client's upload capacity of 60 Mbps. Therefore, the maximum throughput would be 60 Mbps. Alternatively, if the direction was server to client, the server upload is 80, middle link 75, client download 60. So again, 60 is the bottleneck.\n\nBut wait, the problem says \"end-end throughput for each of four client-to-server pairs\". The term \"client-to-server\" might refer to the direction, i.e., data going from client to server. But in that case, the client's upload is 60, middle link share is 75, server's download is 80. So the throughput is the minimum of 60, 75, 80, which is 60. Alternatively, if the pair is considered as server-to-client, the direction would be server upload (80), middle link (75), client download (60). Again, 60 is the minimum. Either way, the bottleneck is the 60 Mbps link.\n\nBut wait, maybe I need to consider that the middle link is the shared resource. Let's think about each link's capacity. Each server has an 80 Mbps link to the hub, and each client has a 60 Mbps link from the hub. The hub's middle link is 300 Mbps. So if all four pairs are transferring data, each would have their own server link (80), their own client link (60), and share the middle link (300/4=75). So each transfer is constrained by the minimum of these three. But since server link is 80, client link is 60, and middle link share is 75, the minimum is 60. Hence, the maximum throughput per pair is 60 Mbps.\n\nAlternatively, perhaps the middle link is the sum of all the connections. Wait, no. The middle link is a shared resource. If all four pairs are going through the middle link, which has a total capacity of 300 Mbps, and it's divided equally, each gets 75 Mbps. So the middle link can only give each pair 75 Mbps. But if the client or server links are lower than that, then they would be the bottleneck.\n\nBut in this case, the client links are 60, which is lower than 75. So the client link is the bottleneck. Therefore, regardless of the middle link's share, the client link's 60 Mbps would limit the throughput.\n\nTherefore, the answer should be 60. But let me verify.\n\nAlternatively, maybe I need to consider the aggregate throughput. But the question asks for the maximum achievable end-end throughput for each of the four pairs. So per pair, it's 60.\n\nBut wait, another perspective: perhaps the server link is 80, client link is 60, and the middle link per pair is 75. So, when sending data from server to client, the server can send at 80, but the middle link only allows 75, and the client can receive at 60. So the actual throughput is the minimum of these. So 60. Similarly, if sending from client to server, client can send at 60, middle link allows 75, server can receive at 80. So again, 60 is the limit.\n\nTherefore, the correct answer should be 60.\n\nWait, but the options given in the problem were not included here. Wait, the user hasn't provided the options. Wait, looking back at the original problem statement, the user says:\n\n\"The options:\n\nPlease reason step by step...\"\n\nWait, actually, the user hasn't included the options here. The original problem statement has options, but in the user's message, they omitted them. So in the original problem (probably from a test or something), there are options, but the user didn't copy them here. However, the user wants me to provide the answer boxed. So I need to figure out the answer without the options.\n\nBut given the description, the answer is 60. Let me confirm once again.\n\nEach client-server pair's throughput is limited by the minimum of the three rates: server link (80), middle link fair share (75), client link (60). The minimum is 60. Hence, the maximum throughput is 60 Mbps. So the answer is 60.\n\nBut wait, let me check another angle. Suppose all four pairs are active. Each is trying to send as much as possible. The server links can each handle 80, the client links each 60, and the middle link gives each pair 75. So the server can send 80, but the middle link only takes 75 from it, and then the client can only take 60 from the middle link. So the end result is that the client is the bottleneck, receiving at 60. Therefore, the throughput is 60.\n\nYes, that seems correct. So the answer is 60.\n\n**Final Answer**\n\\boxed{60}</think>\n\n<answer>The problem involves determining the maximum achievable end-to-end throughput for each of four client-to-server pairs in a network topology. The key components are:\n\n1. **Server links (RS):** Each server is connected to the central hub with a transmission capacity of 80 Mbps.\n2. **Client links (RC):** Each client is connected to the central hub with a transmission capacity of 60 Mbps.\n3. **Middle link (R):** The central hub has a transmission capacity of 300 Mbps, which is fairly shared among the four pairs, giving each pair 75 Mbps.\n\nThe end-to-end throughput for each pair is determined by the minimum capacity in the path. The three relevant capacities for each pair are:\n- Server link (80 Mbps)\n- Middle link fair share (75 Mbps)\n- Client link (60 Mbps)\n\nThe bottleneck is the minimum of these three values, which is 60 Mbps. Therefore, the maximum achievable end-to-end throughput for each client-to-server pair is:\n\n\\[\n\\boxed{60}\n\\]</answer>",
        "role": "assistant"
    }
]
```

- Final answer:
```text
<answer> ... Therefore, the maximum achievable end-to-end throughput for each client-to-server pair is:

\[
\boxed{60}
\]</answer>
```

- Usage:
```json
{
    "completion_tokens": 2155, 
    "prompt_tokens": 558, 
    "total_tokens": 2713
}
```


### 2. 多图推理
- image_1:


<div style="text-align:center;">
  <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/mmmu_r1/validation_Art_Theory_27_1.png" alt="image" width="600">
</div>

- image_2:

<div style="text-align:center;">
  <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/mmmu_r1/validation_Art_Theory_27_2.png" alt="image" width="600">
</div>


- Id: 
```text
validation_Art_Theory_27
```

- Question:
```text
Who was the patron of the building? <image_1> <image_2>
```

- Options:
```text
['Abd al-Malik', 'Justinian', 'Charlemagne', 'Shah Jahan']
```


- Answer:
```text
B
```

- Messages: (with DeepSeek-R1 model generated reasoning content)
```json
[
    {
        "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
        "role": "system"
    },
    {
        "content": "Refer to following question, description of image and given options: \n\nThe question:\nWho was the patron of the building? <image_1> <image_2>\n\nThe description of the image_1:\nThe image is a detailed architectural floor plan of a large, complex building, likely a cathedral or church, given its intricate design and symmetrical layout. The background is a solid reddish-orange color, while the outlines and details of the structure are depicted in black.\n\n### Key Features:\n\n1. **Central Dome Structure**:\n   - At the center of the plan is a prominent circular area, which appears to be the base of a dome. This central dome is surrounded by four smaller domes or semi-circular apses, creating a cross-like arrangement.\n   - The central dome is connected to the surrounding areas through arched openings, suggesting a spacious interior with high ceilings.\n\n2. **Nave and Aisles**:\n   - Extending from the central dome are long rectangular sections that form the nave and aisles of the building. These sections are lined with smaller rectangular spaces, likely representing pews or seating areas for congregants.\n   - The nave is flanked by two rows of smaller rectangular rooms or chapels, which are connected by corridors.\n\n3. **Transept**:\n   - Perpendicular to the nave is a transept, forming a cross shape when combined with the nave. The transept also features smaller domes or apses at its ends, mirroring the design of the central dome.\n\n4. **Entrance and Vestibule**:\n   - On the left side of the plan, there is a large rectangular area that serves as the main entrance or vestibule. This area is connected to the nave and has additional smaller rooms branching off from it.\n   - The entrance area includes a rounded section, possibly an apse or a baptistery, and a series of steps leading up to the main door.\n\n5. **Side Chapels and Rooms**:\n   - Along the perimeter of the building, there are numerous smaller rectangular rooms, likely used as side chapels, confessionals, or storage areas.\n   - These rooms are arranged in a systematic manner, with some connected directly to the main corridors and others accessible through smaller passageways.\n\n6. **Bell Tower and Staircases**:\n   - In the upper right corner of the plan, there is a tall, narrow structure that appears to be a bell tower. It is connected to the main building through a small corridor.\n   - There are several staircases indicated by diagonal lines, suggesting multiple levels within the building.\n\n7. **Decorative Elements**:\n   - The plan includes various decorative elements such as columns, arches, and ornamental patterns, which add to the grandeur and complexity of the design.\n   - The use of different line weights and styles helps to differentiate between structural elements and decorative features.\n\nOverall, the floor plan depicts a highly detailed and symmetrical design, characteristic of large religious buildings with a rich architectural heritage. The central dome, cross-shaped layout, and numerous side chapels all contribute to the building's majestic and sacred appearance.\n\nThe description of the image_2:\nThis image depicts the interior of a grand, historic building with intricate architectural and decorative elements. The structure appears to be a mosque or a former cathedral, as suggested by its design and features.\n\n### Key Elements:\n\n1. **Dome and Ceiling:**\n   - The central feature is a large dome at the top, adorned with detailed patterns and geometric designs in gold and blue tones.\n   - Smaller domes are visible beneath the main one, each decorated similarly with ornate motifs.\n   - The ceiling has a starburst pattern radiating from the center of the dome, adding to the visual complexity.\n\n2. **Windows and Lighting:**\n   - Numerous arched windows line the upper walls, allowing natural light to filter into the space.\n   - The light creates a warm ambiance, highlighting the golden hues of the decorations.\n\n3. **Walls and Decorations:**\n   - The walls are covered with elaborate mosaics, frescoes, and calligraphy, predominantly in gold, blue, and green tones.\n   - Arabic inscriptions are prominently displayed on circular medallions hanging from the walls, indicating Islamic influence.\n\n4. **Arches and Columns:**\n   - The building features multiple levels of arches supported by columns, creating an open and spacious feel.\n   - The arches are intricately designed, with some displaying floral and geometric patterns.\n\n5. **Furniture and Fixtures:**\n   - A wooden pulpit (minbar) is located on the right side, used for sermons or speeches.\n   - A chandelier hangs from the ceiling, providing additional lighting.\n   - There are prayer rugs and other furnishings scattered across the floor, suggesting the space is used for worship.\n\n6. **People:**\n   - A group of people is gathered near the lower left corner, possibly engaged in prayer or a religious ceremony.\n   - Their presence adds a sense of scale and activity to the otherwise serene environment.\n\n7. **Overall Atmosphere:**\n   - The combination of light, color, and intricate details creates a majestic and spiritual atmosphere, reflecting the cultural and historical significance of the building.\n\nThe image captures the harmony between architecture, art, and spirituality, showcasing the beauty and craftsmanship of this iconic structure.\n\n The options:\nA: Abd al-Malik\nB: Justinian\nC: Charlemagne\nD: Shah Jahan\n\n\nPlease reason step by step, and put your final answer (best option) within \\boxed{}.",
        "role": "user"
    },
    {
        "content": "<think>Okay, let's tackle this question. The question is asking who the patron of the building shown in the images is, with the options being A: Abd al-Malik, B: Justinian, C: Charlemagne, D: Shah Jahan. \n\nFirst, I need to analyze the descriptions of both images to figure out which historical figure is associated with such a structure. \n\nStarting with image_1: It's a floor plan of a large religious building, likely a cathedral or church, with a central dome, transept, nave, and aisles. The description mentions it's symmetrical and has a cross-like layout with a central dome surrounded by smaller domes. The presence of a bell tower in the upper right corner suggests it might be a Christian church, maybe a cathedral. But image 2's description complicates things a bit. \n\nImage 2 shows the interior of a grand building with a large dome, Arabic inscriptions, mosaics with geometric patterns, a minbar (pulpit), prayer rugs, and people engaged in what seems like prayer. The mention of Arabic calligraphy and Islamic features like the minbar and prayer rugs points towards this being a mosque. However, the description also notes it could be a former cathedral, which makes me think of a structure that was converted from a church to a mosque or vice versa. \n\nThe combination of a central dome, cross-shaped plan (from image 1), and Islamic decorations (image 2) is interesting. Let me think about famous buildings that have such features. The Hagia Sophia comes to mind. Originally a Christian cathedral built under Emperor Justinian in Constantinople (now Istanbul), it was later converted into a mosque after the Ottoman conquest. The Hagia Sophia has a massive central dome, a cross layout, and when it became a mosque, Islamic elements like minarets, calligraphy, and a minbar were added. \n\nLooking at the options:\n\nA. Abd al-Malik: He was an Umayyad caliph who built the Dome of the Rock in Jerusalem. The Dome of the Rock is an Islamic shrine with a golden dome, but its structure is different from Hagia Sophia. It's an octagonal building, not a cross-shaped one with a central dome and transept. The floor plan described here doesn't match the Dome of the Rock.\n\nB. Justinian: He was the Byzantine Emperor who commissioned the Hagia Sophia. If the building in question is Hagia Sophia, then Justinian would be the patron. The original structure was a cathedral, which fits the cross-shaped floor plan with a central dome. Later conversion into a mosque explains the Islamic elements in image 2.\n\nC. Charlemagne: He was a Frankish king and Holy Roman Emperor. His reign included the Carolingian Renaissance, and he built the Palatine Chapel in Aachen. That chapel has a central octagonal structure but is quite different from the described layout. The floor plan here is more complex and larger, fitting a major cathedral or mosque, not a palace chapel.\n\nD. Shah Jahan: He was the Mughal emperor who built the Taj Mahal. The Taj Mahal is a mausoleum with a large central dome and four minarets, but the floor plan described here doesn't match. The Taj Mahal is symmetrical but not cross-shaped with a transept and nave. Also, the interior described in image 2 with Arabic calligraphy could fit, but the Taj Mahal isn't a mosque converted from a church. Its primary function is as a tomb.\n\nPutting this together, the Hagia Sophia, built under Justinian (option B), fits both the architectural layout described in image 1 (Christian cathedral features) and the Islamic elements in image 2 (after its conversion). The patron here would be Justinian, as he commissioned the original building. Even though later Islamic patrons added elements, the question asks for the patron of the building, which refers to the original construction. \n\nAlternatively, if the building is a mosque from the start with such a cross-shaped plan, that might be different. But cross-shaped plans are more typical of Christian churches. The presence of a transept and nave in image_1 strongly suggests a Christian origin, later adapted into a mosque. Therefore, the original patron would be Justinian. \n\nWait, but image 2's description says \"former cathedral,\" which aligns with Hagia Sophia's history. So yes, the answer should be B: Justinian.</think>\n\n<answer>The architectural descriptions and historical context point to the Hagia Sophia. Image 1's cross-shaped floor plan with a central dome, transept, and nave aligns with its original design as a Byzantine cathedral commissioned by Emperor Justinian. Image 2's Islamic elements, such as Arabic calligraphy and a minbar, reflect its later conversion into a mosque. The **patron of the original building** is Justinian, who oversaw its construction. \n\n\\boxed{B}</answer>",
        "role": "assistant"
    }
]
```

- Final answer:
```text
<answer>... The **patron of the original building** is Justinian, who oversaw its construction. \n\n\\boxed{B}</answer>
```

- Usage:
```json
{
    "completion_tokens": 1021, 
    "prompt_tokens": 1146, 
    "total_tokens": 2167
}
```



## 数据结构


| 字段   | 类型     | 描述                                  | 样例                                                                                                     |
|------|--------|-------------------------------------|--------------------------------------------------------------------------------------------------------|
| id   | string | 唯一id                                | "validation_Accounting_3"                                                                              |
| question | string | 问题                                  | "... <image 1> Which project should be chosen if the appropriate discount rate is 15 percent?"         |
| options | list   | 选项                                  | ['Project A', 'Project B']                                                                             |
| explanation | string | 补充释义                                | "Discount each project's ..."                                                                          |
| image_1 | dict   | 图1                                  | {'bytes': xx, 'path': 'validation_Accounting_3_1.png', 'description': 'xxx'}                           |
| image_2 | dict   | 图2 （可能为空）                           |                                                                                                        |
| image_3 | dict   | 图3 （可能为空）                           |                                                                                                        |
| image_4 | dict   | 图3 （可能为空）                           |                                                                                                        |
| image_5 | dict   | 图3 （可能为空）                           |                                                                                                        |
| image_6 | dict   | 图3 （可能为空）                           |                                                                                                        |
| image_7 | dict   | 图3 （可能为空）                           |                                                                                                        |
| img_type     | list   | 图形类型                                | ['Tables']                                                                                             |
| answer | string | 答案                                  | "B"                                                                                                    |
| topic_difficulty | string | 题目难度                                | "Medium"                                                                                               |
| question_type | string | 问题类型                                | "multiple-choice"                                                                                      |
| subfield | string | 子领域                                 | "Managerial Accounting"                                                                                |
| source | string | 数据来源                                | "MMMU"                                                                                                 |
| messages | list   | 与R1会话                               | 如下文示例                                                                                                  |
| usage | dict   | token消耗信息                           | 如下文示例                                                                                                  |
| images | list   | 图像聚合，将上述image_1 ~ image_7，聚合为一个list | [{'description': 'xxx', 'path': 'https://www.modelscope.cn/xxx/validation_Agriculture_16_1.png'}, ...] |


* `messages`字段示例
```json
[
    {
        "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
        "role": "system"
    },
    {
        "content": "Refer to following question, description of image and given options: \n\nThe question:\n<image 1> What is the common term for the yellow area surrounding the site of an infection?\n\nThe description of the image_1:\nThe image shows a close-up view of a plant growing in soil. The focus is on the leaves, which are large and have a deep green color with visible veins running through them. One of the leaves, prominently displayed in the foreground, exhibits signs of damage or disease. There are distinct patches of discoloration on this leaf, ranging from yellow to brown, indicating possible decay or infection. These discolored areas appear dry and brittle compared to the healthy parts of the leaf.\n\nThe leaf's surface has a slightly rough texture, and there are small specks of dirt or debris scattered across it. The stem of the plant is visible, connecting the leaves together, and it appears sturdy and green. The background consists of dark, rich soil, suggesting that the plant is growing in an outdoor environment. Other leaves can be seen partially in the frame, but they are not as detailed as the main leaf in focus. The overall scene suggests a natural setting where the plant may be experiencing some form of stress or health issue.\n\n The options:\nA: I don't know and I don't want to guess\nB: Corona\nC: Border\nD: Halo\nE: Toxin zone\n\n\nPlease reason step by step, and put your final answer (best option) within \\boxed{}.",
        "role": "user"
    },
    {
        "content": "<think>Okay, let me try to figure this out. The question is asking about the common term for the yellow area surrounding the site of an infection on a plant. The image described shows a plant leaf with yellow to brown discolored patches, possibly indicating decay or infection. The options are B to E, with D being Halo.\n\nHmm, I remember from biology classes that when plants get infected, especially by pathogens like bacteria or fungi, there can be visible symptoms around the infected area. Sometimes, the tissue around the infection site turns yellow. I think this is part of the plant's response to the infection. The term \"halo\" sounds familiar here. Like a halo around the infected spot. Wait, I've heard the term \"chlorotic halo\" before. Chlorosis is the yellowing of plant tissue, so a chlorotic halo would be a yellow ring around the infection. That makes sense. The other options don't seem right. Corona is more like a part of a virus or a structure in some organisms. Border is too generic. Toxin zone doesn't ring a bell as a standard term. So probably D: Halo is the correct answer here.</think>\n\n<answer>The question refers to the yellow area surrounding an infection site on a plant leaf. In plant pathology, a common term for such a yellow ring or zone around a diseased area is a **chlorotic halo**, where chlorosis (yellowing due to loss of chlorophyll) occurs as part of the plant's response to infection. Among the options provided, **D: Halo** directly corresponds to this concept. Other terms like \"Border\" or \"Toxin zone\" are not standard, and \"Corona\" is unrelated. Thus, the best answer is:\n\n\\boxed{D}</answer>",
        "role": "assistant"
    }
]
```

* `usage`字段示例
```json
{
  "completion_tokens":359,
  "prompt_tokens":308,
  "total_tokens":667
}
```


## 数据合成

数据合成采用`Perception-Reasoning`流水线，即图像描述生成和自然语言推理来合成数据。具体步骤如下：
1. 使用Qwen2.5-VL-72B-Instruct进行图像理解并生成详细的图像描述
2. 使用DeepSeek-R1生成自然语言推理会话


### prompt构建

* **Caption生成**
```text
Please use english to describe the image in detail, including the objects, actions, and relationships that everything you see in the image.
```
* **Reasoning content生成**

  - system prompt:
    ```text
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    ```

  - user prompt template:
    ```text
    \n\nPlease reason step by step, and put your final answer (best option) within \\boxed{}.
    ```


## 使用方式


**MMMU-Reasoning-Distill-Validation** 包含839条内置Reasoning messages的样本（**未校验准确答案**），均来自MMMU数据集的验证集；数据集包含1个jsonl格式文件(含有captions, messages, image url等), 以及一个文件夹(916 images)存放所有图片，总大小357MB。


### 1. 使用modelscope命令行下载

```shell
pip install modelscope -U 

modelscope download ModelScope/MMMU-Reasoning-Distill-Validation --repo-type dataset --local_dir ./MMMU-Reasoning-Distill-Validation
```


### 2. 使用modelscope sdk下载

```shell
pip install modelscope[framework] -U
```

```python
from modelscope import MsDataset

ds = MsDataset.load("ModelScope/MMMU-Reasoning-Distill-Validation", split='validation', trust_remote_code=False)
sample = next(iter(ds))

print(sample)
```



## 后续计划

1. 在此数据集基础上，探索多模态GRPO训练在不同类型和复杂度的任务、不同模型上的表现（进行中）
2. 对比多模态GRPO训练和SFT的性能、效率等指标（进行中）
3. Qwen2.5-VL模型蒸馏，目标是在多模态复杂Reasoning/Planning/Understanding任务上达到同尺寸模型性能最佳
4. 开源代码，包括数据合成、模型训练、评估等


## 引用

```text
@misc {ModelScope_Team_2025,
  author       = { {The ModelScope Team} },
  title        = { MMMU-Reasoning-Distill-Validation },
  year         = 2025,
  url          = { https://modelscope.cn/datasets/modelscope/MMMU-Reasoning-Distill-Validation },
  publisher    = { ModelScope }
}
```
