package opencvl

import (
	"errors"
	"image"

	"github.com/Nv7-Github/go-cl"
)

type openCLLayer struct {
	prog       *cl.Program
	ctx        *cl.Context
	kernelName string
	args       []interface{}

	kernel *cl.Kernel
	kargs  []interface{}
	queue  *cl.CommandQueue
	d      *cl.Device
}

// NewOpenCLLayer creates a new OpenCL layer
func NewOpenCLLayer(program string, kernelName string, platform, device int) (Layer, error) {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, err
	}

	p := platforms[platform]

	devices, err := p.GetDevices(cl.DeviceTypeAll)
	if err != nil {
		return nil, err
	}

	d := devices[device]

	ctx, err := cl.CreateContext([]*cl.Device{d})
	if err != nil {
		return nil, err
	}

	prog, err := ctx.CreateProgramWithSource([]string{program})
	if err != nil {
		return nil, err
	}

	err = prog.BuildProgram(nil, "")
	if err != nil {
		return nil, err
	}

	return &openCLLayer{
		prog:       prog,
		ctx:        ctx,
		kernelName: kernelName,
		args:       make([]interface{}, 0),
		d:          d,
	}, nil
}

func (l *openCLLayer) Type() string {
	return "opencl"
}

func (l *openCLLayer) build() error {
	queue, err := l.ctx.CreateCommandQueue(l.d, 0)
	if err != nil {
		return err
	}
	kernel, err := l.prog.CreateKernel(l.kernelName)
	if err != nil {
		return err
	}

	l.queue = queue
	l.kernel = kernel
	return nil
}

func (l *openCLLayer) setArgs(progArgs []interface{}) error {
	args := make([]interface{}, 0)
	for _, progArg := range progArgs {
		switch v := progArg.(type) {
		case int32:
			args = append(args, v)
			break
		case uint32:
			args = append(args, v)
			break
		case float32:
			args = append(args, v)
			break
		case int:
			args = append(args, int32(v))
			break
		case int64:
			args = append(args, int32(v))
			break
		default:
			return errors.New("Invalid Argument Type")
		}
	}
	l.args = args
	return nil
}

func (l *openCLLayer) execute(img *image.RGBA, time int) (*image.RGBA, error) {
	rect := img.Rect
	stride := img.Stride
	bounds := img.Bounds()

	format := cl.ImageFormat{
		ChannelOrder:    cl.ChannelOrderRGBA,
		ChannelDataType: cl.ChannelDataTypeUNormInt8,
	}
	desc := cl.ImageDescription{
		Type:     cl.MemObjectTypeImage2D,
		Width:    bounds.Dx(),
		Height:   bounds.Dy(),
		RowPitch: stride,
	}
	clImg, err := l.ctx.CreateImage(cl.MemReadOnly|cl.MemCopyHostPtr, format, desc, img.Pix)
	if err != nil {
		return nil, err
	}

	out, err := l.ctx.CreateImageFromImage(cl.MemWriteOnly|cl.MemCopyHostPtr, image.NewRGBA(rect))
	if err != nil {
		return nil, err
	}

	args := append([]interface{}{clImg, out, int32(time)}, l.args...)
	err = l.kernel.SetArgs(args...)
	if err != nil {
		return nil, err
	}

	local, err := l.kernel.PreferredWorkGroupSizeMultiple(l.d)
	if err != nil {
		return nil, err
	}

	xSize := bounds.Dx()
	diff := xSize % local
	xSize += local - diff

	ySize := bounds.Dy()
	diff = ySize % local
	ySize += local - diff

	_, err = l.queue.EnqueueNDRangeKernel(l.kernel, nil, []int{xSize, ySize}, []int{local, local}, nil)
	if err != nil {
		return nil, err
	}

	err = l.queue.Finish()
	if err != nil {
		return nil, err
	}

	final := image.NewRGBA(rect)
	final.Stride = stride
	_, err = l.queue.EnqueueReadImage(out, true, [3]int{0, 0, 0}, [3]int{final.Bounds().Dx(), final.Bounds().Dy(), 1}, stride, 0, final.Pix, nil)
	if err != nil {
		return nil, err
	}
	return final, nil
}
