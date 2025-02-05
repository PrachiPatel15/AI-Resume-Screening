import streamlit as st
import pandas as pd
import PyPDF2
import json
from typing import Dict, Any, List, Optional 
from qa_system import ResumeQASystem
from utils import groq_generate 
import streamlit.components.v1 as components

def read_resume(uploaded_file) -> str:
    """Extract text from uploaded resume file"""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    else:
        text = uploaded_file.getvalue().decode()
    return text

def parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse the LLM response with improved work experience handling and certificates section"""
    sections = {
       "Basic Info": {  # Changed to nested structure for basic info
            "Name": "",
            "Email": "",
            "Phone": ""
        },
        "Profile Summary": "",
        "Work Experience": [],  # Changed to list to store multiple experiences
        "Education": "",
        "Technical Skills": "",
        "Projects": [],
        "Certificates": ""  # Added certificates section
    }
    
    section_markers = {
        "name:": ("Basic Info", "Name"),
        "email:": ("Basic Info", "Email"),
        "phone:": ("Basic Info", "Phone"),
        "phone number:": ("Basic Info", "Phone"),
        "profile summary:": ("Profile Summary", None),
        "summary:": ("Profile Summary", None),
        "work experience:": ("Work Experience", None),
        "employment:": ("Work Experience", None),
        "education:": ("Education", None),
        "technical skills:": ("Technical Skills", None),
        "skills:": ("Technical Skills", None),
        "projects:": ("Projects", None),
        "certificates:": ("Certificates", None),
        "certifications:": ("Certificates", None)
    }
    
    lines = text.split('\n')
    current_section = None
    current_subsection = None
    section_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_lower = line.lower()
        
        # Check if this line starts a new section
        new_section = None
        new_subsection = None
        for marker, (section, subsection) in section_markers.items():
            if line_lower.startswith(marker):
                new_section = section
                new_subsection = subsection
                content = line[len(marker):].strip()
                break
        
        if new_section:
            # Save content from previous section if it exists
            if current_section and section_content:
                if current_section == "Work Experience":
                    sections[current_section] = parse_work_experience('\n'.join(section_content))
                elif current_section == "Projects":
                    sections[current_section] = parse_projects('\n'.join(section_content))
                elif current_section == "Basic Info" and current_subsection:
                    sections[current_section][current_subsection] = '\n'.join(section_content).strip()
                else:
                    sections[current_section] = '\n'.join(section_content).strip()
            
            # Start new section
            current_section = new_section
            current_subsection = new_subsection
            section_content = [content] if content else []
        elif current_section:
            # Add line to current section
            section_content.append(line)
    
    # Save the last section's content
    if current_section and section_content:
        if current_section == "Work Experience":
            sections[current_section] = parse_work_experience('\n'.join(section_content))
        elif current_section == "Projects":
            sections[current_section] = parse_projects('\n'.join(section_content))
        elif current_section == "Basic Info" and current_subsection:
            sections[current_section][current_subsection] = '\n'.join(section_content).strip()
        else:
            sections[current_section] = '\n'.join(section_content).strip()
    
    return sections

def parse_work_experience(text: str) -> List[Dict[str, Any]]:
    """Parse work experience with improved company and responsibility detection"""
    experiences = []
    current_exp = None
    current_responsibilities = []
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check for new company entry (starts with bullet or contains date in parentheses)
        if (line.startswith(('•', '-', '*', '○', '·', '►', '▪', '➢')) and 
            ('(' in line and ')' in line)) or \
           (not line.startswith(('•', '-', '*', '○', '·', '►', '▪', '➢')) and 
            '(' in line and ')' in line):
            
            # Save previous experience if exists
            if current_exp and current_responsibilities:
                current_exp['responsibilities'] = current_responsibilities
                experiences.append(current_exp)
            
            # Clean up the line
            if line.startswith(('•', '-', '*', '○', '·', '►', '▪', '➢')):
                line = line[1:].strip()
            
            # Parse company info
            try:
                company_part, date_part = line.split('(', 1)
                date_part = date_part.rstrip(')')
                
                # Split location if present
                if ',' in date_part:
                    date_info, location = date_part.rsplit(',', 1)
                else:
                    date_info, location = date_part, ""
                
                current_exp = {
                    'company': company_part.strip(),
                    'duration': date_info.strip(),
                    'location': location.strip(),
                    'responsibilities': []
                }
                current_responsibilities = []
            except ValueError:
                # Handle malformed lines
                current_exp = {
                    'company': line,
                    'duration': '',
                    'location': '',
                    'responsibilities': []
                }
                current_responsibilities = []
        
        # Check for responsibility
        elif line.startswith(('•', '-', '*', '○', '·', '►', '▪', '➢')) and current_exp:
            resp = line.lstrip('•-*○·►▪➢ ').strip()
            if resp:
                current_responsibilities.append(resp)
        
        i += 1
    
    # Add final experience
    if current_exp and current_responsibilities:
        current_exp['responsibilities'] = current_responsibilities
        experiences.append(current_exp)
    
    return experiences

def parse_projects(text: str) -> List[Dict[str, Any]]:
    """Parse projects into structured format"""
    projects = []
    current_project = None
    current_details = []
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if line starts with bullet and might be a project title
        if line.startswith(('•', '-', '*', '○', '·', '►', '▪', '➢')):
            clean_line = line.lstrip('•-*○·►▪➢ ').strip()
            
            # If current line doesn't contain "Technologies:" and next lines have bullets,
            # it's likely a project title
            is_title = True
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith(('•', '-', '*', '○', '·', '►', '▪', '➢')):
                    if "technologies:" in clean_line.lower() or \
                       "developed" in clean_line.lower() or \
                       "implemented" in clean_line.lower() or \
                       "built" in clean_line.lower():
                        is_title = False
            
            if is_title:
                # Save previous project if exists
                if current_project and current_details:
                    current_project['details'] = current_details
                    projects.append(current_project)
                
                # Start new project
                current_project = {
                    'title': clean_line,
                    'details': []
                }
                current_details = []
            elif current_project:
                current_details.append(clean_line)
        
        i += 1
    
    # Add final project
    if current_project and current_details:
        current_project['details'] = current_details
        projects.append(current_project)
    
    return projects

def extract_info(resume_text: str) -> Dict[str, Any]:
    """Extract information with improved prompt for projects"""
    prompt = f"""
    Please analyze the following resume and extract the information in this exact format with clear section headers:

    Name: [Full Name]
    Email: [Email Address]
    Phone: [Phone Number]
    Profile Summary: [Detailed profile summary]
    Work Experience: [List each position in this format:
    • Company Name (Duration, Location)
    * Responsibility 1
    * Responsibility 2
    * Responsibility 3
    ]
    Education: [Detailed education history]
    Technical Skills: [List of technical skills]
    Projects: [List each project in this format:
    * Project Title
    * Detail 1
    * Detail 2
    * Technologies: List of technologies used
    ]
    Certificates: [List of certificates and certifications]

    Important: 
    - For Work Experienc or experience, first try to find company name and then find bullet points below it. Maintain the bullet point format exactly as shown above. Do this for every company you can find.
    - For Projects, ensure each project title is on its own line with a bullet point, followed by details on separate lines

    Resume Text: {resume_text}
    """
    
    raw_extracted_text = groq_generate(prompt)
    parsed_data = parse_llm_response(raw_extracted_text)
    return parsed_data

def display_section_content(section: str, data: Dict[str, Any]):
    """Display section content with combined basic info"""
    if section not in data:
        st.write("No data available for this section.")
        return

    content = data[section]
    if not content:
        st.write(f"No information available for {section}")
        return

    if section == "Basic Info":
        st.write("**Basic Information:**")
        for field in ["Name", "Email", "Phone"]:
            if content.get(field):
                st.write(f"**{field}:** {content[field]}")
    elif section == "Work Experience":
        st.write("**Work Experience:**")
        for exp in content:
            st.write(f"\n**{exp['company']}**")
            st.write(f"*{exp['duration']} | {exp['location']}*")
            for resp in exp['responsibilities']:
                st.write(f"• {resp}")
    elif section == "Projects":
        st.write("**Projects:**")
        for project in content:
            st.write(f"\n**{project['title']}**")
            for detail in project['details']:
                st.write(f"• {detail}")
    else:
        st.write(f"**{section}:**")
        st.write(content)

def main():

    st.set_page_config(page_title="📄 AI Resume Screening", layout="wide")

    if "qa_system" not in st.session_state:
        st.session_state.qa_system = ResumeQASystem()
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = None
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None

    with st.sidebar:
        st.header("📂 Upload Resume")
        uploaded_file = st.file_uploader("Choose a resume", type=['pdf', 'txt'])
        if uploaded_file:
            st.session_state.resume_text = read_resume(uploaded_file)
            st.session_state.extracted_data = extract_info(st.session_state.resume_text)
            st.session_state.qa_system.create_knowledge_base(st.session_state.resume_text)
            st.success("✅ Resume processed successfully!")

    st.header("🕵️ AI Detective: Investigate This Resume")
    tab1, tab2 = st.tabs(["🤖 AI-Powered Analysis", "📜 Resume Breakdown"])

    with tab2:
        if st.session_state.extracted_data:
            sections = ["Basic Info", "Profile Summary", "Work Experience", "Education", 
                        "Technical Skills", "Projects", "Certificates"]
            selected_section = st.selectbox("📌 Select Section:", sections)
            display_section_content(selected_section, st.session_state.extracted_data)
        else:
            st.warning("⚠️ Please upload a resume first.")

    with tab1:
        if st.session_state.resume_text:
            
            st.markdown(
                """
                🤖 Ask AI anything about skills, experience, projects, and qualifications.  
                🏆 Get precise answers tailored for hiring decisions.
                """
            )

            # Question input (prevent "Enter" submission)
            question = st.text_input("💬 Ask a Question:", key="question_input", help="Type your question and click 'Get Answer'")

            # ✅ Centering button and using a form to prevent Enter submission
            with st.form("qa_form", clear_on_submit=False):
                submitted = st.form_submit_button("🚀 Get Answer", use_container_width=True)

            # ✅ Answer display logic
            if submitted and question:
                with st.spinner("🤖 Thinking..."):
                    answer = st.session_state.qa_system.answer_question(question)

                # ✅ Answer box with single border
                st.markdown(
                    f"""
                    <div style="
                        border-left: 4px solid #4CAF50;
                        padding: 12px;
                        border-radius: 8px;
                        background-color: #1e1e1e;
                        color: white;
                        margin-top: 10px;
                        font-size: 16px;
                        line-height: 1.6;
                    ">
                        <b>📄 Answer:</b><br> {answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 🎨 **Improved Styling (Centered, Single Border)**
            st.markdown(
                """
                <style>
                    .stTextInput input { font-size: 16px; padding: 10px; border-radius: 10px; }
                    .custom-button { padding: 8px 20px; font-size: 14px; border-radius: 6px; background-color: #e74c3c; color: white; border: none; cursor: pointer; }
                    .custom-button:hover { background-color: #c0392b; }
                    .stMarkdown { font-size: 18px; line-height: 1.6; }
                </style>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.warning("⚠️ Please upload a resume first.")


if __name__ == "__main__":
    main()
