import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import os

os.makedirs("outputs", exist_ok=True)

# Load Data
topics = pd.read_csv("data/topic_labeled.csv")
summaries = pd.read_csv("data/ai_topic_summaries.csv")

# Create Visualization for Report
plt.figure(figsize=(8,5))
topics["topic"].value_counts().plot(kind="bar", color="lightblue")
plt.title("Topic Distribution in Amazon Reviews")
plt.xlabel("Topic ID")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/report_topic_distribution.png")
plt.close()

# Build PDF Report
styles = getSampleStyleSheet()
doc = SimpleDocTemplate("outputs/Customer_Insights_Report.pdf", pagesize=A4)
story = []

story.append(Paragraph("<b>Customer Review Insights Report</b>", styles["Title"]))
story.append(Spacer(1, 20))
story.append(Image("outputs/report_topic_distribution.png", width=400, height=250))
story.append(Spacer(1, 20))

for _, row in summaries.iterrows():
    story.append(Paragraph(f"<b>Topic {int(row['topic'])}</b>", styles["Heading2"]))
    story.append(Paragraph(row["summary"], styles["BodyText"]))
    story.append(Spacer(1, 10))

doc.build(story)
print("Report saved to outputs/Customer_Insights_Report.pdf")
