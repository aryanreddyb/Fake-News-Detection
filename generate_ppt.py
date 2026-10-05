import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def apply_font(run, size=18, bold=False):
    run.font.name = 'Arial'
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0) # Black

def add_slide(prs, title_text, content_bullets=None, notes_text=""):
    slide_layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    
    # Set Title
    title_shape = slide.shapes.title
    title_shape.text = title_text
    title_shape.text_frame.paragraphs[0].font.name = 'Arial'
    title_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 0, 0)
    title_shape.text_frame.paragraphs[0].font.bold = True
    
    # Set Content
    if content_bullets:
        body_shape = slide.shapes.placeholders[1]
        tf = body_shape.text_frame
        tf.clear() # Clear default
        for i, bullet in enumerate(content_bullets):
            p = tf.add_paragraph()
            p.text = bullet
            p.font.name = 'Arial'
            p.font.size = Pt(20)
            p.font.color.rgb = RGBColor(0, 0, 0)
            if i > 0:
                p.space_before = Pt(10)
                
    # Add Notes
    if notes_text:
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = notes_text
        
    return slide

def create_presentation():
    prs = Presentation()
    
    # Slide 1: Title Slide
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "Multimodal & Graph-Enhanced Fake News Detection"
    title.text_frame.paragraphs[0].font.name = 'Arial'
    title.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 0, 0)
    
    subtitle.text = "End-to-End Pipeline, Methodology, and Analysis"
    subtitle.text_frame.paragraphs[0].font.name = 'Arial'
    subtitle.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 0, 0)
    
    # Notes for Slide 1
    slide.notes_slide.notes_text_frame.text = "Layman terms: This presentation covers our project on detecting fake news. Instead of just looking at the text (like most systems), we also look at how posts are connected to each other (like who posted them and what they link to).\nKey terminology: Multimodal, Graph Neural Networks (GNN), Baseline Model, Fake News Detection."

    # Slide 2: Introduction
    add_slide(prs, "Problem Statement & Introduction", 
              ["• Fake news doesn't exist in a vacuum; it spreads through networks.",
               "• Traditional NLP (Natural Language Processing) models only look at the text, which can be fooled easily.",
               "• Solution: Combine text understanding with social context (authors, links).",
               "• We use Graph Neural Networks to map these connections and improve detection accuracy."],
              notes_text="Layman terms: We are explaining that reading the text isn't enough to catch fake news anymore. We need to see who is talking to who, and how information spreads. Our solution combines reading the text with mapping the network of posts.\nKey terminologies: NLP (Natural Language Processing), Social Context, Graph Neural Networks (GNN).")

    # Slide 3: Key Terminologies
    add_slide(prs, "Key Terminologies",
              ["• Graph Neural Network (GNN): A model that learns from data structured as a graph (nodes and edges).",
               "• Node / Edge: A node represents a post. An edge represents a connection (shared author/link) between posts.",
               "• ModernBERT: A state-of-the-art transformer model used to understand the text (Semantic Embedding).",
               "• GraphSAGE: A specific type of GNN that samples and aggregates information from a node's neighbors.",
               "• Embeddings: Dense vectors (lists of numbers) representing the meaning of text.",
               "• Epochs: One complete pass through the training dataset."],
              notes_text="Layman terms: These are the technical words the professor might ask about. Nodes are the posts, edges are the lines connecting them. ModernBERT is the 'reader' that understands text. GraphSAGE is the 'network analyzer' that looks at the connections. Embeddings are just mathematical translations of text into numbers so the computer can process it.\nKey terminologies: GNN, Node, Edge, ModernBERT, GraphSAGE, Embeddings, Epochs.")

    # Slide 4: Tech Stack
    add_slide(prs, "Technology Stack",
              ["• Language / Core: Python 3",
               "• Deep Learning Framework: PyTorch",
               "• Graph Processing: PyTorch Geometric (PyG)",
               "• NLP / Transformers: HuggingFace Transformers (ModernBERT)",
               "• Data Manipulation: Pandas, NumPy",
               "• Evaluation Metrics: Scikit-learn (Accuracy, F1-Score)"],
              notes_text="Layman terms: This is the software and tools we used to build the project. PyTorch is the main engine for our AI models. PyTorch Geometric specifically handles the graph math. HuggingFace provides the ModernBERT text model.\nKey terminologies: PyTorch, PyTorch Geometric (PyG), HuggingFace, Transformers, Scikit-learn.")

    # Slide 5: Project Pipeline Diagram
    slide = add_slide(prs, "Project Pipeline & Methodology", [], 
                      notes_text="Layman terms: This diagram shows the flow of our data. First we clean it, then we use ModernBERT to read the text and turn it into numbers (embeddings). At the same time, we build a map of how the posts are connected (Graph Construction). Finally, GraphSAGE combines both to make the final prediction (Fake or Real).\nKey terminologies: Data Preprocessing, Text Encoding, Embeddings, Graph Construction, GraphSAGE.")
    
    # Draw simple pipeline
    shapes = slide.shapes
    width = Inches(1.8)
    height = Inches(0.8)
    
    b1 = shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(2), width, height)
    b1.text = "Raw Data\nPreprocessing"
    b1.fill.solid()
    b1.fill.fore_color.rgb = RGBColor(255, 255, 255)
    b1.line.color.rgb = RGBColor(0, 0, 0)
    for p in b1.text_frame.paragraphs: p.font.color.rgb = RGBColor(0, 0, 0)

    b2 = shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3), Inches(1.5), width, height)
    b2.text = "Text Encoding\n(ModernBERT)"
    b2.fill.solid()
    b2.fill.fore_color.rgb = RGBColor(255, 255, 255)
    b2.line.color.rgb = RGBColor(0, 0, 0)
    for p in b2.text_frame.paragraphs: p.font.color.rgb = RGBColor(0, 0, 0)

    b3 = shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3), Inches(2.7), width, height)
    b3.text = "Graph Builder\n(Nodes & Edges)"
    b3.fill.solid()
    b3.fill.fore_color.rgb = RGBColor(255, 255, 255)
    b3.line.color.rgb = RGBColor(0, 0, 0)
    for p in b3.text_frame.paragraphs: p.font.color.rgb = RGBColor(0, 0, 0)

    b4 = shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.5), Inches(2), width + Inches(0.5), height)
    b4.text = "GraphSAGE\n(Feature + Structure)"
    b4.fill.solid()
    b4.fill.fore_color.rgb = RGBColor(255, 255, 255)
    b4.line.color.rgb = RGBColor(0, 0, 0)
    for p in b4.text_frame.paragraphs: p.font.color.rgb = RGBColor(0, 0, 0)

    b5 = shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.5), Inches(2), width, height)
    b5.text = "Classification\n(Real / Fake)"
    b5.fill.solid()
    b5.fill.fore_color.rgb = RGBColor(255, 255, 255)
    b5.line.color.rgb = RGBColor(0, 0, 0)
    for p in b5.text_frame.paragraphs: p.font.color.rgb = RGBColor(0, 0, 0)

    # Arrows (simple lines)
    shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(2.4), Inches(2.3), Inches(0.5), Inches(0.2)).fill.solid()
    shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.9), Inches(1.8), Inches(0.5), Inches(0.2)).fill.solid()
    shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.9), Inches(3.0), Inches(0.5), Inches(0.2)).fill.solid()
    shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(7.9), Inches(2.3), Inches(0.5), Inches(0.2)).fill.solid()

    # Slide 6: Step 1 & 2 - Setup and Data
    add_slide(prs, "Step 1 & 2: Data Preprocessing & Pilot Dataset",
              ["• Dataset: Fakeddit (Multimodal fake news dataset).",
               "• Preprocessing: Filtered missing metadata (authors/subreddits), cleaned titles, removed empty text rows.",
               "• Pilot Generation: Sampled a balanced 10,000 node dataset (8k train, 1k validation, 1k test) to allow fast experimentation.",
               "• Purpose: High-quality, balanced data is crucial for preventing bias and memory issues during graph training."],
              notes_text="Layman terms: First we took a massive dataset and cleaned up the messy bits (missing authors, blank text). Then we took a smaller, perfectly balanced slice of 10,000 posts so we could test our ideas quickly without crashing the computer's memory.\nKey terminologies: Fakeddit, Preprocessing, Pilot Dataset, Data Balancing, Nodes.")

    # Slide 7: Step 3 - Baseline Model
    add_slide(prs, "Step 3: Baseline Language Model Training",
              ["• Model: ModernBERT (answerdotai/ModernBERT-base).",
               "• Architecture: 2024 SOTA Transformer with FlashAttention-2 and RoPE.",
               "• Goal: Fine-tune the model strictly on the post titles to classify real vs. fake news based only on text.",
               "• Outcome: Achieved an initial baseline accuracy (83.4%) and extracted high-quality text weights."],
              notes_text="Layman terms: We trained a highly advanced AI reader (ModernBERT) to guess if a post is fake just by reading the title. It did pretty well, but we use this as a 'baseline'—the score we want to beat by adding our graph network later.\nKey terminologies: ModernBERT, Fine-tuning, Baseline, FlashAttention-2, RoPE (Rotary Positional Embeddings).")

    # Slide 8: Step 4 - Graph Construction
    add_slide(prs, "Step 4: Relational Graph Construction",
              ["• Process: Treat the 10,000 posts as nodes in a graph.",
               "• Edges created based on explicit links (linked_submission_id) and implicit communities (shared authors).",
               "• Constraint: Capped author connections at 50 posts to prevent 'bot' accounts from causing out-of-memory errors (degree explosion).",
               "• Result: A graph with 10,000 nodes and 9,878 directed edges representing the social network."],
              notes_text="Layman terms: We built a web showing how posts relate. If two posts have the same author, we draw a line between them. If one post replies to another, we draw a line. We stopped super-active accounts (like bots) from drawing too many lines so the computer wouldn't crash.\nKey terminologies: Graph Construction, Nodes, Edges, Directed Edges, Degree Explosion, Adjacency Matrix.")

    # Slide 9: Step 5 & 6 - Embedding & GNN
    add_slide(prs, "Step 5 & 6: Feature Extraction & GraphSAGE",
              ["• Embedding Extraction: Pass all 10,000 posts through ModernBERT to extract a [10000, 768] feature matrix.",
               "• GNN Training: Train GraphSAGE (Sample and Aggregate) on this matrix + the graph structure.",
               "• Algorithm: GraphSAGE samples neighbors and aggregates their features to update the target node's representation.",
               "• Leakage Prevention: Used strict transductive masks (train_mask, val_mask, test_mask) to ensure the test data remained completely unseen during training."],
              notes_text="Layman terms: We turned all 10,000 texts into lists of 768 numbers. Then GraphSAGE looks at those numbers, plus the numbers of the post's 'neighbors' in the web we built, to make a final decision. We hid the test data carefully so the model couldn't cheat.\nKey terminologies: Feature Extraction, Embeddings (768-dimensional), GraphSAGE, Transductive Masks, Data Leakage, Aggregation.")
              
    # Slide 10: Comparative Analysis & Metrics
    add_slide(prs, "Comparative Analysis & Final Metrics",
              ["• Baseline Model (Text-Only ModernBERT):",
               "   - Validation Accuracy: 83.40%",
               "   - Validation F1-Score: 0.7883",
               "• Final Model (GraphSAGE - Text + Social Graph):",
               "   - Validation Accuracy: 85.60%  |  Validation F1-Score: 0.8159",
               "   - Test Accuracy (Unseen): 86.20%",
               "   - Test F1-Score (Unseen): 0.8333",
               "• Conclusion: Adding graph context (who posts and links) improved F1 score by ~4.5%."],
              notes_text="Layman terms: This is the proof our idea worked. The text-only AI got a score of 0.78. When we added the network of who-talks-to-who, the score jumped to 0.83 on brand new data it had never seen before. It proves fake news is easier to catch when you look at the social connections.\nKey terminologies: Validation Accuracy, Test Accuracy, F1-Score, Text-Only Baseline, Graph Context, Generalization.")

    # Slide 11: Key Obstacles
    add_slide(prs, "Key Obstacles & Solutions",
              ["• Memory Limits (Degree Explosion): Highly active bot accounts created massive cliques (fully connected subgraphs) that crashed RAM. Solution: Capped max edges per author at 50.",
               "• Data Leakage in Graphs: GNNs can accidentally 'see' test labels through edges. Solution: Applied transductive masking (train/val/test masks) to hide test labels during loss calculation.",
               "• Computing Heavy NLP: Training transformers is slow. Solution: Used ModernBERT with FlashAttention to drastically speed up processing and extracted embeddings offline before GNN training."],
              notes_text="Layman terms: We hit three big problems. 1. Spammers made our web too dense and crashed the computer, so we limited them. 2. In networks, it's easy for the AI to accidentally cheat by looking at its neighbors' answers, so we strictly hid the test answers (masks). 3. Text AI is slow, so we used the newest fast model and split the work into two stages.\nKey terminologies: Degree Explosion, Cliques, Data Leakage, Transductive Masking, Offline Embedding Extraction.")

    # Slide 12: Future Work (What can be done)
    add_slide(prs, "Future Work (What Can Be Done Next)",
              ["• Scale to Full Dataset: Run the validated pipeline on the complete 560,000+ node dataset.",
               "• Image/Multimodal Integration: Incorporate the 'image_url' and 'hasImage' data. Combine image embeddings (e.g., using ResNet/CLIP) with text embeddings before GNN passing.",
               "• Edge Weights: Assign different weights to edges (e.g., a linked submission edge might be stronger than a shared author edge).",
               "• Alternative Architectures: Experiment with GATv2 (Graph Attention Networks) to dynamically learn which neighbors are most suspicious."],
              notes_text="Layman terms: What's next? First, run this on the massive half-million post dataset. Second, start looking at the pictures in the posts, not just the text. Third, we can teach the AI that some connections (like direct replies) matter more than others (like having the same author).\nKey terminologies: Scaling, Multimodal Integration, Image Embeddings, Edge Weights, GATv2 (Graph Attention Networks).")

    # Slide 13: Final Takeaway
    add_slide(prs, "Final Takeaway",
              ["• Text Alone is Insufficient: Modern deceptive content mimics authentic writing styles effectively.",
               "• Structure Matters: The social context—who is posting, and how they link to others—contains critical signals for detecting misinformation.",
               "• Hybrid Approach Wins: Fusing semantic embeddings (ModernBERT) with topological structure (GraphSAGE) successfully yields a highly accurate, scalable fake news detection system (86.2% Test Accuracy)."],
              notes_text="Layman terms: The main lesson is that you can't just read a post to know if it's fake anymore. You have to look at the network around it. By combining advanced reading AI with advanced network AI, we built a very strong system that catches fake news 86% of the time.\nKey terminologies: Semantic Embeddings, Topological Structure, Hybrid Architecture, Scalability.")

    # Save
    prs.save('d:/Capstone/fnd/Fake_News_Detection_Presentation.pptx')
    print("PPTX saved successfully.")

if __name__ == '__main__':
    create_presentation()
