run_trainer:
	python3 training_runner.py

test_authentication:
	curl \
		-X POST \
		-H "Content-Type: application/json" \
		-d '{"email": "testmail@fyp.com", "password": "1234321"}' \
		http://localhost:8000/api/token/ \
		# > output.txt

test_create_user_api:
	curl \
		-X POST \
		-H "Content-Type: application/json" \
		-d '{"email": "testmail@fyp.com", "password": "1234321", "firstname": "bob", "lastname": "ross"}' \
		http://localhost:8000/api/user/create_user/ \
		> output.txt