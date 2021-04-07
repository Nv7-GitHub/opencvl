__kernel void solver(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int time)
	{
		const int2 pos = (int2)(get_global_id(0), get_global_id(1));
		const int2 dim = get_image_dim(in);
		float4 pixel = (float4)(0);
		float4 pix;
		int nbs = 0;
		int2 off;
		if (pos.x < dim.x && pos.y < dim.y) {
			pixel = read_imagef(in, pos);

			if (pixel.g == pixel.b && pixel.b == 0) {
				write_imagef(out, pos, pixel);
				return;
			}

			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					if (!(x == y && x == 0) && !(abs(x) == 1 && abs(y) == 1)) {
						off = pos+(int2)(x, y);
						if (off.x >= 0 && off.x <= dim.x && off.y >= 0 && off.y <= dim.y) {
							pix = read_imagef(in, off);
							if (pix.r == 1) {
								nbs++;
							}
						}
					}
				}
			}
			if (nbs <= 1) {
				pixel = (float4)(0);
			}

			pixel.a = 1;
			write_imagef(out, pos, pixel);
		}
	}