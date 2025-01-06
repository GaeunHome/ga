# Genetic Algorithm
<h3>Problem definition</h3>
<p>
In a class, the professor assigns students to work on group presentations. Each group consists of M students, and the presentation requires completing N different tasks. Each task has a varying time requirement—some tasks take longer to complete, while others are relatively simple. Additionally, each student has different strengths for various tasks; some are skilled in writing reports, while others excel in creating slides or delivering presentations.<br>
The professorʼs objectives are:<br>
1. <b>Every student must be assigned at least one task</b>, ensuring that all members contribute to the preparation process.<br>
2. <b>Assign each task to the most suitable student</b>, minimizing the overall time required for the group to complete the entire project.
</p>

# Example
<p>
In a course on Charisma Studies, Shinny and Jack are team members grouped according to their MBTI types, both being ENTJs. ENTJs are natural-born leaders who exude charisma and confidence, projecting an aura of authority and rallying people toward a common goal. For their final project on Charisma Studies, they plan to demonstrate how to confidently approach strangers without creating an awkward situation. Therefore, they have divided the work into three tasks: approaching strangers, filming and editing the video, and creating the report.
Shinny, being lively and outgoing, is not afraid to approach strangers, so it only takes her 3 hours to complete this task. Jack, on the other hand, is a bit shy and reserved, needing 5 hours for the same task. However, Jack has a passion for photography and can efficiently handle filming and editing in just 2 hours, while Shinny, in contrast, would need 8 hours. When it comes to creating the report, Shinny is more adept and can produce a high-quality report in just 6 hours, which is 1 hour less than Jack would need to complete the same task.
</p>
<h3>Solution</h3>
<p>According to the problem statement, there are a total of two students and three tasks. Each student requires different amounts of time to complete each task.
Therefore, it can be represented as follows:<br>
<img width="442" alt="截圖 2025-01-06 下午4 11 11" src="https://github.com/user-attachments/assets/9d4683f2-9b7b-42cf-866a-28bea0840a1a" /><br>
After calculating using the genetic algorithm, the following results were obtained:<br>
<img width="442" alt="截圖 2025-01-06 下午4 11 49" src="https://github.com/user-attachments/assets/4d10e161-27b1-4a36-8329-8316d00874fb" /><br>
This indicates that assigning Shinny to approach strangers and create the report, while Jack handles filming and editing, can minimize the total time required, completing all tasks in just 11 hours! This also illustrates the saying, "the capable should take on more responsibility."
</p>
