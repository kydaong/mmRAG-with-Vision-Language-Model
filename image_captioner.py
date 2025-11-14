"""
Image captioner using Claude Sonnet 4.5 (Vision)
This script done after "doc_processor.ipynb" is tested 
"""
import anthropic
import base64
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv



class ImageCaptioner:
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv() #load .env variables

        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        for var in proxy_vars:
            os.environ.pop(var, None)

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY") 
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or parameters")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"


        
    def encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Convert image to base64 and detect media type
        Returns: (base64_string, media_type)
        """
        with open(image_path, "rb") as image_file:
            image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
        
        # Detect media type from extension
        suffix = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(suffix, 'image/jpeg')
        
        return image_data, media_type
    

    
    def caption_image(self, image_path: str, context: str = "") -> str:
        """
        Generate detailed technical caption for Oil & Gas equipment/diagrams
        
        Args:
            image_path: Path to image file
            context: Additional context (document name, page number, etc.)
        
        Returns:
            Detailed technical description
        """
        image_data, media_type = self.encode_image(image_path)
        
        # ⚠️ IMPORTANT: Detailed prompt is critical for good results
        prompt = f"""You are analyzing a technical image from an Oil & Gas industrial document.

Provide a detailed, structured description for technical retrieval.

**Analyze and describe:**

1. **Document Type**: P&ID, equipment photo, schematic, flowchart, safety diagram, etc.

2. **Equipment Visible**: 
   - List all equipment with tag numbers if visible (pumps, valves, tanks, compressors, heat exchangers)
   - Equipment types and specifications

3. **Piping & Flow**:
   - Flow directions
   - Pipe connections
   - Line sizes if visible

4. **Instrumentation & Controls**:
   - Sensors, transmitters, controllers (PT, TT, FT, LT, FV, PV, etc.)
   - Tag numbers
   - Control loops

5. **Safety Equipment**:
   - Relief valves, safety interlocks
   - Warning symbols
   - Emergency systems

6. **Text & Labels**:
   - All visible text, labels, tag numbers
   - Process conditions (pressure, temperature, flow rates)
   - Equipment specifications

7. **Condition Assessment** (if equipment photo):
   - Visible damage, corrosion, wear
   - Maintenance status
   - Anomalies

{f'**Context**: {context}' if context else ''}

Use industry-standard terminology (API, ASME, ISA)."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.2,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            caption = response.content[0].text
            return caption
            
        except anthropic.APIError as e:
            print(f"Anthropic API error: {e}")
            return f"Error: API error - {str(e)}"
        except Exception as e:
            print(f"Error captioning image: {e}")
            return f"Error: Could not generate caption - {str(e)}"



if __name__ == "__main__":
    captioner = ImageCaptioner()
    # Test on first few images
    image_dir = Path("data/processed/images")
    #image_dir = Path("D:/Projects/mmRAG-with-Vision-Language-Model/notebook/data/processed/images")
    
    if image_dir.exists():
        images = list(image_dir.glob("*.*"))[:3]  # Test first 3 images
    
        
        if images:
            print("Testing Claude image captioner...\n")
            print("Using Claude Sonnet 4.5 for technical image analysis")
            print("="*60)
            
            for img_path in images:
                print(f"\nImage: {img_path.name}")
                print("="*60)
                
                caption = captioner.caption_image(
                    str(img_path),
                    context=f"Source document: {img_path.stem}"
                )
                
                print(caption)
                print()
        else:
            print("No images found in image file")
    else:
        print("Images directory not found. Run document_processor first.")
