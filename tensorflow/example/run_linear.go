package main

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {

	x := float32(4)

	model, err := tf.LoadSavedModel("linearmodel", []string{"serve"}, nil)
	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}
	defer model.Session.Close()

	inputTensor, err := tf.NewTensor([][]float32{[]float32{x}})
	if err != nil {
		fmt.Printf("Error creating input tensor: %s\n", err.Error())
		return
	}

	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("input").Output(0): inputTensor,
		},
		[]tf.Output{
			model.Graph.Operation("pred").Output(0),
		},
		nil,
	)

	if err != nil {
		fmt.Printf("Error running the session, err: %s\n", err.Error())
		return
	}

	fmt.Printf("The value of y = %v when x = %v \n", result[0].Value(), x)
}
