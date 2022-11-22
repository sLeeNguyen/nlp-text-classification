setup: requirements.txt
		pip install -r requirements.txt

dev:
		uvicorn main:app --reload

clean:
		rm -rf __pycache__