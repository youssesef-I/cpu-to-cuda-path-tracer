#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <memory>
#include <limits>
#include <cstdlib>

inline double random_double() {
    return std::rand() / (RAND_MAX + 1.0);
}
inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}

class vec3 {
public:
    double e[3];
    vec3() : e{0, 0, 0} {}
    vec3(double x, double y, double z) : e{x, y, z} {}
    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }
    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    vec3& operator+=(const vec3& v) {
        e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2];
        return *this;
    }
    vec3& operator*=(double t) {
        e[0] *= t; e[1] *= t; e[2] *= t;
        return *this;
    }
    double length() const { return std::sqrt(length_squared()); }
    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
};
inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]);
}
inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0]-v.e[0], u.e[1]-v.e[1], u.e[2]-v.e[2]);
}
inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
inline vec3 operator*(const vec3& v, double t) { return t * v; }
inline double dot(const vec3& u, const vec3& v) {
    return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2];
}
inline vec3 unit_vector(const vec3& v) { return v * (1.0 / v.length()); }

using point3 = vec3;
using color  = vec3;

// --- random vector helpers: must come AFTER vec3, since they return vec3 ---

// A random vector with each component in [min, max).
inline vec3 random_vec3(double min, double max) {
    return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}
// Rejection sampling: draw from the unit cube, keep only points inside the
// unit sphere, then normalize. Cheap uniform directions with no trig.
// NOTE: the unpredictable loop count is GPU-hostile -- a phase-3 target.
inline vec3 random_unit_vector() {
    while (true) {
        vec3 p = random_vec3(-1, 1);
        double lensq = p.length_squared();
        if (lensq > 1e-160 && lensq <= 1.0)
            return p * (1.0 / std::sqrt(lensq));
    }
}
// A random direction in the hemisphere facing the same way as the normal.
inline vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    return (dot(on_unit_sphere, normal) > 0.0) ? on_unit_sphere : -on_unit_sphere;
}

class ray {
public:
    point3 orig;
    vec3 dir;
    ray() {}
    ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction) {}
    point3 origin() const { return orig; }
    vec3 direction() const { return dir; }
    point3 at(double t) const { return orig + t * dir; }
};

struct hit_record {
    point3 p;
    vec3 normal;
    double t;
    bool front_face;
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    virtual ~hittable() = default;
    virtual bool hit(const ray& r, double ray_tmin, double ray_tmax,
                     hit_record& rec) const = 0;
};

class sphere : public hittable {
public:
    point3 center;
    double radius;
    sphere(const point3& c, double r) : center(c), radius(r) {}
    bool hit(const ray& r, double ray_tmin, double ray_tmax,
             hit_record& rec) const override {
        vec3 oc = r.origin() - center;
        double a = r.direction().length_squared();
        double half_b = dot(oc, r.direction());
        double c = oc.length_squared() - radius * radius;
        double discriminant = half_b*half_b - a*c;
        if (discriminant < 0) return false;
        double sqrtd = std::sqrt(discriminant);
        double root = (-half_b - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (-half_b + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root) return false;
        }
        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) * (1.0 / radius);
        rec.set_face_normal(r, outward_normal);
        return true;
    }
};

class hittable_list : public hittable {
public:
    std::vector<std::shared_ptr<hittable>> objects;
    hittable_list() {}
    void add(std::shared_ptr<hittable> object) { objects.push_back(object); }
    bool hit(const ray& r, double ray_tmin, double ray_tmax,
             hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = ray_tmax;
        for (const auto& object : objects) {
            if (object->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
};

// Path tracing: a ray bounces until it escapes to the sky or runs out of
// depth. The sky IS the light source -- there is no explicit lamp.
color ray_color(const ray& r, int depth, const hittable& world) {
    if (depth <= 0)
        return color(0, 0, 0);            // out of bounces: gather no more light

    hit_record rec;
    // tmin = 0.001, not 0: a bounced ray starts *on* the surface, and floating-
    // point error would otherwise let it immediately re-hit the surface it just
    // left ("shadow acne").
    if (world.hit(r, 0.001, std::numeric_limits<double>::infinity(), rec)) {
        vec3 direction = random_on_hemisphere(rec.normal);
        return 0.5 * ray_color(ray(rec.p, direction), depth - 1, world);
    }

    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

// Displays expect gamma-encoded values; without this the image looks too dark.
inline double linear_to_gamma(double linear) {
    return (linear > 0) ? std::sqrt(linear) : 0;
}

int main() {
    const double aspect_ratio = 16.0 / 9.0;
    const int width = 1200;
    const int samples_per_pixel = 500;
    const int max_depth = 10;
    int height = int(width / aspect_ratio);
    height = (height < 1) ? 1 : height;

    hittable_list world;
    world.add(std::make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(std::make_shared<sphere>(point3(0, -100.5, -1), 100));

    double focal_length = 1.0;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(width) / height);
    point3 camera_center = point3(0, 0, 0);
    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);
    vec3 pixel_delta_u = viewport_u * (1.0 / width);
    vec3 pixel_delta_v = viewport_v * (1.0 / height);
    point3 viewport_upper_left = camera_center
        - vec3(0, 0, focal_length) - viewport_u * 0.5 - viewport_v * 0.5;
    point3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    std::ofstream out("image.ppm");
    out << "P3\n" << width << ' ' << height << "\n255\n";

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                double px = -0.5 + random_double();
                double py = -0.5 + random_double();
                point3 pixel_sample = pixel00_loc
                    + ((double(i) + px) * pixel_delta_u)
                    + ((double(j) + py) * pixel_delta_v);
                vec3 ray_direction = pixel_sample - camera_center;
                ray r(camera_center, ray_direction);
                pixel_color += ray_color(r, max_depth, world);
            }
            pixel_color *= (1.0 / samples_per_pixel);
            double r_out = linear_to_gamma(pixel_color.x());
            double g_out = linear_to_gamma(pixel_color.y());
            double b_out = linear_to_gamma(pixel_color.z());
            int ir = int(256 * (r_out < 0 ? 0 : (r_out > 0.999 ? 0.999 : r_out)));
            int ig = int(256 * (g_out < 0 ? 0 : (g_out > 0.999 ? 0.999 : g_out)));
            int ib = int(256 * (b_out < 0 ? 0 : (b_out > 0.999 ? 0.999 : b_out)));
            out << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}
