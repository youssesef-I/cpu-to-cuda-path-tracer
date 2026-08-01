#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <curand_kernel.h>

// ===========================================================================
// GPU ray tracer -- naive (correct, not-yet-optimized) CUDA port.
//   * math types are dual-callable (__host__ __device__)
//   * scene is a flat array of concrete spheres (no virtuals, no shared_ptr)
//   * ray_color is ITERATIVE (recursion flattened into an attenuation loop)
//   * per-thread randomness via cuRAND
// Phase 3 optimization applied: warp-divergence removed from the RNG
// (analytic unit vector). Further tuning (-arch=sm_89, block size) is
// done via compiler flags / the tx,ty constants below.
// ===========================================================================

#define CUDA_CHECK(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        std::cerr << "CUDA error = " << (unsigned)result << " at "
                  << file << ":" << line << " '" << func << "' \n"
                  << cudaGetErrorString(result) << "\n";
        cudaDeviceReset();
        exit(99);
    }
}

// ------------------------------- vec3 --------------------------------------
class vec3 {
public:
    float e[3];
    __host__ __device__ vec3() : e{0, 0, 0} {}
    __host__ __device__ vec3(float x, float y, float z) : e{x, y, z} {}
    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }
    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2]; return *this;
    }
    __host__ __device__ vec3& operator*=(float t) {
        e[0] *= t; e[1] *= t; e[2] *= t; return *this;
    }
    __host__ __device__ float length() const { return sqrtf(length_squared()); }
    __host__ __device__ float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
};
__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]);
}
__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0]-v.e[0], u.e[1]-v.e[1], u.e[2]-v.e[2]);
}
__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
__host__ __device__ inline vec3 operator*(const vec3& v, float t) { return t * v; }
__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]);   // component-wise (color * color)
}
__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2];
}
__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v * (1.0f / v.length());
}
using point3 = vec3;
using color  = vec3;

// ------------------------------- ray ---------------------------------------
class ray {
public:
    point3 orig; vec3 dir;
    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction) {}
    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }
    __host__ __device__ point3 at(float t) const { return orig + t * dir; }
};

// --------------------------- geometry --------------------------------------
struct hit_record {
    point3 p; vec3 normal; float t; bool front_face;
    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct sphere {
    point3 center; float radius;
    __device__ bool hit(const ray& r, float ray_tmin, float ray_tmax,
                        hit_record& rec) const {
        vec3 oc = r.origin() - center;
        float a = r.direction().length_squared();
        float half_b = dot(oc, r.direction());
        float c = oc.length_squared() - radius * radius;
        float discriminant = half_b*half_b - a*c;
        if (discriminant < 0.0f) return false;
        float sqrtd = sqrtf(discriminant);
        float root = (-half_b - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (-half_b + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root) return false;
        }
        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) * (1.0f / radius);
        rec.set_face_normal(r, outward_normal);
        return true;
    }
};

// Test a ray against every sphere; keep the nearest hit. Replaces hittable_list.
__device__ bool hit_world(const sphere* spheres, int count,
                          const ray& r, float ray_tmin, float ray_tmax,
                          hit_record& rec) {
    hit_record temp_rec; bool hit_anything = false; float closest_so_far = ray_tmax;
    for (int k = 0; k < count; ++k) {
        if (spheres[k].hit(r, ray_tmin, closest_so_far, temp_rec)) {
            hit_anything = true; closest_so_far = temp_rec.t; rec = temp_rec;
        }
    }
    return hit_anything;
}

// --------------------------- randomness (cuRAND) ---------------------------
// Analytic uniform point on the unit sphere (Archimedes' theorem: a uniform z
// plus a uniform azimuth gives a uniform sphere point). No rejection loop, so
// every thread in a warp does identical work -- this removes the warp
// divergence the old cube-rejection version caused. (Phase-3 optimization.)
__device__ inline vec3 random_unit_vector(curandState* state) {
    float z = 1.0f - 2.0f * curand_uniform(state);          // z in [-1, 1]
    float r = sqrtf(fmaxf(0.0f, 1.0f - z*z));
    float phi = 2.0f * 3.14159265f * curand_uniform(state);
    return vec3(r * cosf(phi), r * sinf(phi), z);
}
__device__ inline vec3 random_on_hemisphere(curandState* state, const vec3& normal) {
    vec3 v = random_unit_vector(state);
    return (dot(v, normal) > 0.0f) ? v : -v;
}

// ------------------- iterative path tracing (no recursion) -----------------
__device__ color ray_color(ray r, const sphere* spheres, int count,
                           int max_depth, curandState* state) {
    color attenuation(1.0f, 1.0f, 1.0f);       // running light multiplier
    for (int depth = 0; depth < max_depth; ++depth) {
        hit_record rec;
        if (hit_world(spheres, count, r, 0.001f, FLT_MAX, rec)) {
            vec3 direction = random_on_hemisphere(state, rec.normal);
            attenuation *= 0.5f;               // 50% grey diffuse surface
            r = ray(rec.p, direction);         // bounce and continue
        } else {
            vec3 unit_direction = unit_vector(r.direction());
            float a = 0.5f * (unit_direction.y() + 1.0f);
            color sky = (1.0f - a) * color(1.0f, 1.0f, 1.0f)
                      + a * color(0.5f, 0.7f, 1.0f);
            return attenuation * sky;          // escaped: fold in the sky
        }
    }
    return color(0, 0, 0);                      // hit the depth cap: no light
}

// ------------------------------- kernel ------------------------------------
__global__ void render(float* fb, int width, int height,
                       int samples_per_pixel, int max_depth,
                       point3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v,
                       point3 camera_center, const sphere* spheres, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curandState state;
    curand_init(1984 + pixel_index, 0, 0, &state);   // per-pixel independent stream

    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        float px = -0.5f + curand_uniform(&state);
        float py = -0.5f + curand_uniform(&state);
        point3 pixel_sample = pixel00_loc
            + ((float(i) + px) * pixel_delta_u)
            + ((float(j) + py) * pixel_delta_v);
        vec3 ray_direction = pixel_sample - camera_center;
        ray r(camera_center, ray_direction);
        pixel_color += ray_color(r, spheres, count, max_depth, &state);
    }
    pixel_color *= (1.0f / samples_per_pixel);

    int fb_index = pixel_index * 3;
    fb[fb_index + 0] = sqrtf(pixel_color.x());   // gamma-correct in the kernel
    fb[fb_index + 1] = sqrtf(pixel_color.y());
    fb[fb_index + 2] = sqrtf(pixel_color.z());
}

int main() {
    const int width  = 1200;
    const int height = 675;
    const int samples_per_pixel = 500;
    const int max_depth = 10;
    const int tx = 8, ty = 8;

    // Camera / viewport (computed on the host, passed into the kernel).
    float focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * (float(width) / height);
    point3 camera_center(0, 0, 0);
    vec3 viewport_u(viewport_width, 0, 0);
    vec3 viewport_v(0, -viewport_height, 0);
    vec3 pixel_delta_u = viewport_u * (1.0f / width);
    vec3 pixel_delta_v = viewport_v * (1.0f / height);
    point3 viewport_upper_left = camera_center
        - vec3(0, 0, focal_length) - viewport_u * 0.5f - viewport_v * 0.5f;
    point3 pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

    // Scene: managed memory so the host fills it and the device reads it.
    const int num_spheres = 2;
    sphere* spheres;
    CUDA_CHECK(cudaMallocManaged((void**)&spheres, num_spheres * sizeof(sphere)));
    spheres[0].center = point3(0, 0, -1);       spheres[0].radius = 0.5f;
    spheres[1].center = point3(0, -100.5f, -1); spheres[1].radius = 100.0f;

    // Framebuffer.
    int num_pixels = width * height;
    size_t fb_size = num_pixels * 3 * sizeof(float);
    float* fb;
    CUDA_CHECK(cudaMallocManaged((void**)&fb, fb_size));

    dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
    dim3 threads(tx, ty);

    // Time ONLY the kernel with CUDA events -- excludes context init and memory
    // migration, giving the honest compute number to compare against the CPU.
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start); cudaEventCreate(&t_stop);
    cudaEventRecord(t_start);

    render<<<blocks, threads>>>(fb, width, height, samples_per_pixel, max_depth,
                                pixel00_loc, pixel_delta_u, pixel_delta_v,
                                camera_center, spheres, num_spheres);

    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, t_start, t_stop);
    std::cerr << "kernel time: " << kernel_ms << " ms\n";

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Write PPM (top row first == j ascending).
    std::ofstream out("image.ppm");
    out << "P3\n" << width << ' ' << height << "\n255\n";
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int idx = (j * width + i) * 3;
            float r = fb[idx + 0], g = fb[idx + 1], b = fb[idx + 2];
            r = r < 0 ? 0 : (r > 0.999f ? 0.999f : r);
            g = g < 0 ? 0 : (g > 0.999f ? 0.999f : g);
            b = b < 0 ? 0 : (b > 0.999f ? 0.999f : b);
            out << int(256*r) << ' ' << int(256*g) << ' ' << int(256*b) << '\n';
        }
    }

    CUDA_CHECK(cudaFree(fb));
    CUDA_CHECK(cudaFree(spheres));
}
