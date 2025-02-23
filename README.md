# MedScan

## Inspiration  
The inspiration for this project came from the increasing role of AI in healthcare. With the growing need for quick and accurate diagnoses, we wanted to build a tool that could assist healthcare professionals by leveraging AI to analyze medical images and provide insights. Additionally, we wanted to integrate a chat feature that would allow doctors to ask AI-driven questions about medical conditions.  

## What We Learned  
Through this project, we deepened my understanding of several key technologies:  
- **Flask:** Used as the backbone for the web application, helping manage routes and API calls.  
- **Bootstrap, HTML, and CSS:** Designed a simple yet functional interface for healthcare professionals.  
- **TorchXRayVision:** Learned how to integrate a pretrained deep learning model for medical image analysis.  
- **Hugging Face Qwen 2:** Implemented a large language model (LLM) to assist doctors in answering medical-related questions.  
- **Model Deployment:** Gained experience in handling AI model inference in a web application.  

## How We Built It  
1. **Setting Up the Backend:** Used Flask to create the web server and handle API requests for AI-powered image analysis and the LLM chat feature.  
2. **Integrating AI Models:**  
   - Used **TorchXRayVision** to process medical images and provide diagnostic predictions.  
   - Integrated **Hugging Face Qwen 2** to power the chatbot for answering medical-related queries.  
3. **Building the Frontend:** Developed a responsive UI with Bootstrap, HTML, and CSS to ensure ease of use.  
4. **Connecting Everything:** Implemented API calls between the frontend and backend to allow seamless interactions with the AI models.  

## Challenges Faced  
- **Model Inference Speed:** Running AI models on a web server required optimization to ensure fast responses for both image analysis and the chatbot.  
- **Handling Large Image Files:** Managing medical image uploads efficiently while maintaining performance was a key challenge.  
- **Fine-Tuning AI Responses:** Ensuring that the chatbot provided relevant and accurate medical-related answers required careful prompt engineering and testing.  

## Conclusion  
This project was a great learning experience, combining AI, web development, and healthcare. By leveraging pretrained AI models, we were able to create a tool that could assist doctors in both diagnosing medical images and answering medical questions. Moving forward, we aim to enhance the model’s accuracy, improve real-time performance, and explore potential regulatory compliance for wider adoption in healthcare settings.  

