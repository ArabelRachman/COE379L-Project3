# Startin and Stopping te server:
docker-compose up
docker-compose down

# Getting the summary
(on Windows)
curl http://localhost:5000/summary

(on Linux)
curl http://172.17.0.1:5000/summary

expected response:
{
  "input_shape": [null, 128, 128, 3],
  "model_type": "lenet_alternate",
  "number_of_parameters": 798657,
  "output_shape": [null, 1]
}

# Making requests example
curl -X POST http://172.17.0.1:5000/inference -F image=@test.jpg

expected response:
{
  "prediction": "damage"
}
or
{
  "prediction": "no_damage"
}
