import {assert, drop, int, nonnull, Mat4, Tensor3, Vec3} from './base.js';

//////////////////////////////////////////////////////////////////////////////
// The graphics engine:

class Camera {
  heading: number; // In radians: [0, 2π)
  pitch: number;   // In radians: (-π/2, π/2)
  zoom: number;
  direction: Vec3;
  position: Vec3;

  // Used to smooth out mouse inputs.
  private last_dx: number;
  private last_dy: number;

  // transform = projection * view is the overall model-view-projection.
  private position_for: Vec3;
  private transform_for: Mat4;
  private transform: Mat4;
  private projection: Mat4;
  private view: Mat4;

  constructor(width: int, height: int) {
    this.pitch = 0;
    this.heading = 0;
    this.zoom = 0;
    this.direction = Vec3.from(0, 0, 1);
    this.position = Vec3.create();

    this.last_dx = 0;
    this.last_dy = 0;

    this.position_for = Vec3.create();
    this.transform_for = Mat4.create();
    this.transform = Mat4.create();
    this.projection = Mat4.create();
    this.view = Mat4.create();

    const aspect = height ? width / height : 1;
    Mat4.perspective(this.projection, Math.PI / 4, aspect, 0.1);
  }

  applyInputs(dx: number, dy: number, dscroll: number) {
    // Smooth out large mouse-move inputs.
    const jerkx = Math.abs(dx) > 400 && Math.abs(dx / (this.last_dx || 1)) > 4;
    const jerky = Math.abs(dy) > 400 && Math.abs(dy / (this.last_dy || 1)) > 4;
    if (jerkx || jerky) {
      const saved_x = this.last_dx;
      const saved_y = this.last_dy;
      this.last_dx = (dx + this.last_dx) / 2;
      this.last_dy = (dy + this.last_dy) / 2;
      dx = saved_x;
      dy = saved_y;
    } else {
      this.last_dx = dx;
      this.last_dy = dy;
    }

    let pitch = this.pitch;
    let heading = this.heading;

    // Overwatch uses the same constant values to do this conversion.
    const conversion = 0.066 * Math.PI / 180;
    dx = dx * conversion;
    dy = dy * conversion;

    heading += dx;
    const T = 2 * Math.PI;
    while (this.heading < 0) this.heading += T;
    while (this.heading > T) this.heading -= T;

    const U = Math.PI / 2 - 0.01;
    this.pitch = Math.max(-U, Math.min(U, this.pitch + dy));
    this.heading = heading;

    const dir = this.direction;
    Vec3.set(dir, 0, 0, 1);
    Vec3.rotateX(dir, dir, this.pitch);
    Vec3.rotateY(dir, dir, this.heading);

    // Scrolling is trivial to apply: add and clamp.
    if (dscroll === 0) return;
    this.zoom = Math.max(0, Math.min(10, this.zoom + Math.sign(dscroll)));
  }

  getTransform(): Mat4 {
    Mat4.view(this.view, this.position, this.direction);
    Mat4.multiply(this.transform, this.projection, this.view);
    return this.transform;
  }

  getTransformFor(offset: Vec3): Mat4 {
    Vec3.sub(this.position_for, this.position, offset);
    Mat4.view(this.view, this.position_for, this.direction);
    Mat4.multiply(this.transform_for, this.projection, this.view);
    return this.transform_for;
  }

  setTarget(x: number, y: number, z: number) {
    Vec3.set(this.position, x, y, z);
    Vec3.scaleAndAdd(this.position, this.position, this.direction, -this.zoom);
  }
};

//////////////////////////////////////////////////////////////////////////////

interface GL extends WebGL2RenderingContext {};

class Shader {
  gl: GL;
  program: WebGLProgram;
  uniforms: WebGLActiveInfo[];
  attributes: WebGLActiveInfo[];

  constructor(gl: GL, source: string) {
    this.gl = gl;
    const parts = source.split('#split');
    const vertex = this.compile(parts[0], gl.VERTEX_SHADER);
    const fragment = this.compile(parts[1], gl.FRAGMENT_SHADER);
    this.program = this.link(vertex, fragment);
    this.uniforms = [];
    this.attributes = [];

    const program = this.program;
    const uniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniforms; i++) {
      const info = gl.getActiveUniform(program, i);
      if (!info || this.builtin(info.name)) continue;
      this.uniforms.push(info);
    }
    const attributes = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < attributes; i++) {
      const info = gl.getActiveAttrib(program, i);
      if (!info || this.builtin(info.name)) continue;
      this.attributes.push(info);
    }
  }

  private builtin(name: string): boolean {
    return name.startsWith('gl_') || name.startsWith('webgl_');
  }

  private compile(source: string, type: number): WebGLShader {
    const gl = this.gl;
    const result = nonnull(gl.createShader(type));
    gl.shaderSource(result, `#version 300 es\n${source}`);
    gl.compileShader(result);
    if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(result);
      gl.deleteShader(result);
      throw new Error(`Unable to compile shader: ${info}`);
    }
    return result;
  }

  private link(vertex: WebGLShader, fragment: WebGLShader): WebGLProgram {
    const gl = this.gl;
    const result = nonnull(gl.createProgram());
    gl.attachShader(result, vertex);
    gl.attachShader(result, fragment);
    gl.linkProgram(result);
    if (!gl.getProgramParameter(result, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(result);
      gl.deleteProgram(result);
      throw new Error(`Unable to link program: ${info}`);
    }
    return result;
  }
};

//////////////////////////////////////////////////////////////////////////////

const kFixedShader = `
  uniform mat4 u_transform;
  in vec3 a_position;
  in vec3 a_color;
  out vec3 v_color;

  void main() {
    v_color = a_color;
    gl_Position = u_transform * vec4(a_position, 1.0);
  }
#split
  precision highp float;

  in vec3 v_color;
  out vec4 o_color;

  void main() {
    o_color = vec4(v_color, 1);
  }
`;

interface FixedGeometry {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  indices: Uint32Array;
  uvs: Float32Array;
};

class FixedMesh {
  gl: GL;
  shader: Shader;
  geo: FixedGeometry;
  vao: WebGLVertexArrayObject | null;
  indices: WebGLBuffer | null;
  buffers: WebGLBuffer[];
  position: Vec3;

  constructor(gl: GL, shader: Shader, geo: FixedGeometry) {
    this.gl = gl;
    this.geo = geo;
    this.vao = null;
    this.indices = null;
    this.buffers = [];
    this.shader = shader;
    this.position = Vec3.create();
  }

  draw() {
    this.prepare();
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indices);
    gl.drawElements(gl.TRIANGLES, this.geo.indices.length, gl.UNSIGNED_INT, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
  }

  dispose() {
    const gl = this.gl;
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.indices);
    for (const buffer of this.buffers) gl.deleteBuffer(buffer);
    this.vao = null;
    this.indices = null;
    this.buffers.length = 0;
  }

  setPosition(x: int, y: int, z: int) {
    Vec3.set(this.position, x, y, z);
  }

  private prepare() {
    if (this.vao) return;
    const gl = this.gl;
    const program = this.shader.program;
    this.vao = nonnull(gl.createVertexArray());
    gl.bindVertexArray(this.vao);
    this.prepareAttribute('a_position', this.geo.positions, 3);
    this.prepareAttribute('a_color', this.geo.colors, 3);
    this.prepareIndices(this.geo.indices);
    gl.bindVertexArray(null);
  }

  private prepareAttribute(name: string, data: Float32Array, size: int) {
    const gl = this.gl;
    const buffer = nonnull(gl.createBuffer());
    const location = gl.getAttribLocation(this.shader.program, name);
    if (location < 0) return;

    gl.enableVertexAttribArray(location);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.buffers.push(buffer);
  }

  private prepareIndices(data: Uint32Array) {
    const gl = this.gl;
    const buffer = nonnull(gl.createBuffer());

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    this.indices = buffer;
  }
};

//////////////////////////////////////////////////////////////////////////////

interface Mesh {
  dispose: () => void,
  setPosition: (x: number, y: number, z: number) => void,
};

class Renderer {
  camera: Camera;
  private gl: WebGL2RenderingContext;
  private location: WebGLUniformLocation;
  private shader: Shader;
  private meshes: FixedMesh[];

  constructor(canvas: HTMLCanvasElement) {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    this.camera = new Camera(canvas.width, canvas.height);

    const gl = nonnull(canvas.getContext('webgl2'));
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    this.gl = gl;

    const shader = new Shader(gl, kFixedShader);
    const program = shader.program;
    this.location = nonnull(gl.getUniformLocation(program, 'u_transform'));
    this.shader = shader;
    this.meshes = [];
  }

  addFixedMesh(geo: FixedGeometry): Mesh {
    const result = new FixedMesh(this.gl, this.shader, geo);
    this.meshes.push(result);
    return result;
  }

  render() {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.shader.program);

    const camera = this.camera;
    const location = this.location;
    for (const mesh of this.meshes) {
      const transform = camera.getTransformFor(mesh.position);
      gl.uniformMatrix4fv(location, false, transform);
      mesh.draw();
    }
  }
};

//////////////////////////////////////////////////////////////////////////////

export {Mesh, Renderer};
