from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from utils import groq_generate 

class ResumeQASystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        self.db = None
        
    def create_knowledge_base(self, text: str):
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        self.db = FAISS.from_documents(documents, self.embeddings)
    
    def extract_skills(self, text: str) -> set:
        """Extract skills from text"""
        prompt = f"""Extract only the technical and professional skills from this text. 
        Return them as a comma-separated list.

        Structure:
        1. Skill-1
        2. Skill-2
        3. Skill-3

        Do not add any more information other than skills
        Do not add Education or Awards or Certifications or Experience in this section
        Text: {text}"""
        
        skills_text = groq_generate(prompt)
        return {skill.strip().lower() for skill in skills_text.split(',') if skill.strip()}
    
    def calculate_skill_match(self, required_skills: set, candidate_skills: set) -> float:
        """Calculate the percentage of required skills matched"""
        if not required_skills:
            return 0.0
        matched_skills = required_skills.intersection(candidate_skills)
        return len(matched_skills) / len(required_skills) * 100
    
    def answer_question(self, question: str) -> str:
        if not self.db:
            return "Please process a resume first."
            
        # Get relevant context
        relevant_docs = self.db.similarity_search(question, k=4)
        context = "\n".join(doc.page_content for doc in relevant_docs)
        
        # For role suitability questions, use special handling
        if self._is_role_suitability_question(question):
            return self._evaluate_role_suitability(question, context)
        
        # For other questions, use standard prompt
        prompt = f"""You are an AI assistant helping HR professionals analyze resumes. Answer the following question accurately based ONLY on the information provided in the resume context: {question}

                    Resume Context:
                    {context}

                    Key Instructions:
                    1. ONLY provide information that is explicitly stated in the resume context
                    2. DO NOT make assumptions or infer information not directly stated
                    3. Divide answer in bullet points strictly for more clarity
                    4. If experience is not mentioned in total years, find from the work experience and give final answer in years and months from starting till march 2025.
                    5. If information is not found, clearly state "Not mentioned in the resume"
                    6. If the question is not related to resume, clearly state "I did not understand, can you please re-type question?"

                    Remember:
                    - Keep responses focused and precise
                    - Never invent or assume information"""

        return groq_generate(prompt)
    
    def _is_role_suitability_question(self, question: str) -> bool:
        """Check if question is about role suitability"""
        keywords = ['suitable', 'fit', 'good for', 'qualified for', 'match', 'right for', 
                   'appropriate for', 'good candidate for', 'consider for']
        return any(keyword in question.lower() for keyword in keywords)
    
    def _evaluate_role_suitability(self, question: str, context: str) -> str:
        """Evaluate candidate's suitability for a role"""
        # Extract required skills from question
        prompt_required = f"""Extract the required skills or qualifications mentioned in this question. 
        Return only the technical and professional skills as a comma-separated list.
        Question: {question}"""
        required_skills_text = groq_generate(prompt_required)
        required_skills = {skill.strip().lower() for skill in required_skills_text.split(',') if skill.strip()}
        
        # Extract candidate's skills from resume
        candidate_skills = self.extract_skills(context)
        
        # Calculate match percentage
        match_percentage = self.calculate_skill_match(required_skills, candidate_skills)
        
        # Generate detailed evaluation
        prompt = f"""Analyze the candidate's suitability for the role based on the following information:

                Required Skills: {', '.join(required_skills)}
                Candidate's Skills: {', '.join(candidate_skills)}
                Skill Match: {match_percentage:.1f}%

                Resume Context:
                {context}

                Provide a structured evaluation with:
                1. **My suggestion**: Overall Assessment (Strong Match/Moderate Match/Limited Match) with summary in 2 lines strictly.
                2. Key Matching Skills (3-4 most relevant matches)
                3. Notable Gaps (if any)
                4. Additional Relevant Experience (from resume context)

                Format as bullet points. Be specific and reference only information from the resume."""

        return groq_generate(prompt)