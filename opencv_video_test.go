// +build videos

package opencvl

import (
	"testing"

	"gocv.io/x/gocv"
)

func TestOpenCVLayer_Video(t *testing.T) {
	pipeline := NewPipeline()
	layer := NewOpenCVLayer(func(mat gocv.Mat, args ...interface{}) gocv.Mat {
		gocv.BitwiseNot(mat, &mat)
		return mat
	})
	err := pipeline.AddLayer(layer)
	if err != nil {
		panic(err)
	}

	err = pipeline.ExecuteOnVideo("sample.mp4", "out.mp4")
	if err != nil {
		panic(err)
	}
}
