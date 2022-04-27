use std::cell::UnsafeCell;

extern { fn console_log(value: i32); }

struct RacyCell<T>(UnsafeCell<T>);

unsafe impl<T> Sync for RacyCell<T> {}

impl<T> RacyCell<T> {
  const fn new(value: T) -> RacyCell<T> {
    RacyCell(UnsafeCell::new(value))
  }

  #[inline(always)]
  unsafe fn get(&self) -> &T {
    &*self.0.get()
  }

  #[inline(always)]
  unsafe fn get_mut(&self) -> &mut T {
    &mut *self.0.get()
  }
}

static MASK_DATA: RacyCell<Vec<i32>> = RacyCell::new(vec![]);
static REGISTRY: RacyCell<Registry> = RacyCell::new(Registry::new());
static VOXELS: RacyCell<Tensor3> = RacyCell::new(Tensor3::new());

struct Registry {
  blocks: Vec<Block>,
  facets: Vec<Facet>,
}

impl Registry {
  const fn new() -> Registry {
    Registry { blocks: vec![], facets: vec![] }
  }
}

const NO_MATERIAL: usize = 0;

struct Block {
  facets: [usize; 6],
  opaque: bool,
  solid: bool,
}

struct Facet {
  color: [f32; 4],
  texture: usize,
}

struct Tensor3 {
  data: Vec<u32>,
  shape: [usize; 3],
  stride: [usize; 3],
}

impl Tensor3 {
  const fn new() -> Tensor3 {
    Tensor3 { data: vec![], shape: [0; 3], stride: [0; 3] }
  }
}

#[no_mangle]
pub extern "C" fn register_block(
  f0: usize,
  f1: usize,
  f2: usize,
  f3: usize,
  f4: usize,
  f5: usize,
  opaque: bool,
  solid: bool,
) {
  let registry = unsafe { REGISTRY.get_mut() };
  registry.blocks.push(Block { facets: [f0, f1, f2, f3, f4, f5], opaque, solid })
}

#[no_mangle]
pub extern "C" fn register_facet(
  c0: f32,
  c1: f32,
  c2: f32,
  c3: f32,
  texture: usize,
) {
  let registry = unsafe { REGISTRY.get_mut() };
  registry.facets.push(Facet { color: [c0, c1, c2, c3], texture })
}

#[no_mangle]
pub extern "C" fn allocate_voxels(x: usize, y: usize, z: usize) -> *mut u32 {
  let voxels = unsafe { VOXELS.get_mut() };
  voxels.data.resize(x * y * z, 0);
  voxels.shape = [x, y, z];
  voxels.stride = [1, x, x * y];
  voxels.data.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn mesh() -> usize {
  let mask_data = unsafe { MASK_DATA.get_mut() };
  let registry = unsafe { REGISTRY.get() };
  let voxels = unsafe { VOXELS.get() };
  mesh_impl(mask_data, registry, voxels)
}

fn mesh_impl(mask_data: &mut Vec<i32>, registry: &Registry, voxels: &Tensor3) -> usize {
  //const result = kGeometryData;
  //result.numQuads = 0;

  let Tensor3 {data, shape, stride} = voxels;
  if shape[0] < 2 || shape[1] < 2 || shape[2] < 2 {
    return 0;
  }
  let mut num_quads = 0;

  for d in 0..3 {
    let dir = d * 2;
    let (u, v) = ((d + 1) % 3, (d + 2) % 3);
    let (ld, lu, lv) = (shape[d] - 2, shape[u] - 2, shape[v] - 2);
    let (sd, su, sv) = (stride[d], stride[u], stride[v]);
    let base = su + sv;

    //let mut pos = [0; 3];
    //let mut du = [0; 3];
    //let mut dv = [0; 3];
    //let mut normal = [0; 3];

    let area = lu * lv;
    if mask_data.len() < area {
      mask_data.resize(area, 0);
    }

    for id in 0..ld {
      let mut n_counter = 0;
      for iu in 0..lu {
        let mut index_counter = base + id * sd + iu * su;
        for _iv in 0..lv {
          let (index, n) = (index_counter, n_counter);
          index_counter += sv;
          n_counter += 1;

          // mask[n] is the face between (id, iu, iv) and (id + 1, iu, iv).
          // Its value is the MaterialId to use, times -1, if it is in the
          // direction opposite `dir`.
          //
          // When we enable ambient occlusion, we shift these masks left by
          // 8 bits and pack AO values for each vertex into the lower byte.
          let block0 = *get(data, index) as usize;
          let block1 = *get(data, index + sd) as usize;
          let facing = get_face_dir(registry, block0, block1, dir);

          if facing == 0 { continue; }
          let mask = if facing > 0 {
            *get(&get(&registry.blocks, block0).facets, dir) as i32
          } else {
            -(*get(&get(&registry.blocks, block1).facets, dir + 1) as i32)
          };
          let ao = if facing > 0 {
            pack_ao_mask(registry, data, index + sd, su, sv)
          } else {
            pack_ao_mask(registry, data, index, su, sv)
          };
          mask_data[n] = (mask << 8) | ao;
        }
      }

      n_counter = 0;
      //pos[d] = id;

      for iu in 0..lu {
        let mut iv = 0;
        while iv < lv {
          let n = n_counter;
          let mask = *get(mask_data, n);
          if mask == 0 {
            iv += 1;
            n_counter += 1;
            continue;
          }

          let mut h = 1;
          while h < lv - iv {
            if mask != *get(mask_data, n + h) { break; }
            h += 1;
          }

          let (mut w, mut nw) = (1, n + lv);
          'outer:
          while w < lu - iu {
            for x in 0..h {
              if mask != *get(mask_data, nw + x) { break 'outer; }
            }
            w += 1;
            nw += lv;
          }

          //pos[u] = iu;
          //pos[v] = iv;
          //du[u] = w;
          //dv[v] = h;
          //normal[d] = if mask > 0 { 1 } else { -1 };
          //add_quad(result, d, w, h, mask, pos, du, dv, normal);
          num_quads += 1;

          nw = n;
          for _wx in 0..w {
            for hx in 0..h {
              mask_data[nw + hx] = 0;
            }
            nw += lv;
          }
          iv += h;
          n_counter += h;
        }
      }
    }
  }

  num_quads
}

#[inline(always)]
fn get<T>(vec: &[T], index: usize) -> &T {
  unsafe { &*vec.as_ptr().offset(index as isize) }
}

#[inline(always)]
fn get_face_dir(registry: &Registry, block0: usize, block1: usize, dir: usize) -> i32 {
  if block0 == block1 { return 0; }
  let opaque0 = get(&registry.blocks, block0).opaque;
  let opaque1 = get(&registry.blocks, block1).opaque;
  if opaque0 && opaque1 { return 0; }
  if opaque0 { return 1; }
  if opaque1 { return -1; }

  let material0 = *get(&get(&registry.blocks, block0).facets, dir);
  let material1 = *get(&get(&registry.blocks, block1).facets, dir + 1);
  if material0 == material1 { return 0; }
  if material0 == NO_MATERIAL { return -1; }
  if material1 == NO_MATERIAL { return 1; }
  return 0;
}

#[inline(always)]
fn pack_ao_mask(registry: &Registry, data: &Vec<u32>, ipos: usize, dj: usize, dk: usize) -> i32 {
  let solid = |i: usize| {
    let block = *get(data, i) as usize;
    registry.blocks[block].solid
  };
  let mut a00 = 0; let mut a01 = 0; let mut a10 = 0; let mut a11 = 0;
  if solid(ipos + dj) { a10 += 1; a11 += 1; }
  if solid(ipos - dj) { a00 += 1; a01 += 1; }
  if solid(ipos + dk) { a01 += 1; a11 += 1; }
  if solid(ipos - dk) { a00 += 1; a10 += 1; }

  if a00 == 0 && solid(ipos - dj - dk) { a00 += 1; }
  if a01 == 0 && solid(ipos - dj + dk) { a01 += 1; }
  if a10 == 0 && solid(ipos + dj - dk) { a10 += 1; }
  if a11 == 0 && solid(ipos + dj + dk) { a11 += 1; }

  // Order here matches the order in which we push vertices in addQuad.
  return (a01 << 6) | (a11 << 4) | (a10 << 2) | a00;
}


// import {int, Tensor3, Vec3} from './base.js';
// import {Mesh, Renderer} from './renderer.js';
//
// //////////////////////////////////////////////////////////////////////////////
//
// type BlockId = int & {__type__: 'BlockId'};
// type MaterialId = int & {__type__: 'MaterialId'};
//
// const kNoMaterial = 0 as MaterialId;
// const kEmptyBlock = 0 as BlockId;
//
// interface Material {
//   color: [number, number, number, number],
//   texture: string | null,
//   textureIndex: int,
// };
//
// interface Registry {
//   _solid: boolean[];
//   _opaque: boolean[];
//   getBlockFaceMaterial(id: BlockId, face: int): MaterialId;
//   getMaterialData(id: MaterialId): Material;
// };
//
// //////////////////////////////////////////////////////////////////////////////
//
// interface GeometryData {
//   numQuads: int;
//   quadMaterials: MaterialId[]; // length: n = numQuads
//   positions: number[];         // length: 12n (Vec3 for each vertex)
//   normals: number[];           // length: 12n (Vec3 for each vertex)
//   indices: int[];              // length: 6n  (2 triangles - 6 indices)
//   colors: number[];            // length: 16n (Color4 for each vertex)
//   uvws: number[];              // length: 12n ((u, v, w) for each vertex)
// };
//
// const kGeometryData: GeometryData = {
//   numQuads: 0,
//   quadMaterials: [],
//   positions: [],
//   normals: [],
//   indices: [],
//   colors: [],
//   uvws: [],
// };
//
// const kTmpPos    = Vec3.create();
// const kTmpDU     = Vec3.create();
// const kTmpDV     = Vec3.create();
// const kTmpNormal = Vec3.create();
//
// const kIndexOffsets = {
//   A: [0, 1, 2, 0, 2, 3],
//   B: [1, 2, 3, 0, 1, 3],
//   C: [0, 2, 1, 0, 3, 2],
//   D: [3, 1, 0, 3, 2, 1],
// };
//
// let kMaskData = new Int16Array();
//
// class TerrainMesher {
//   solid: boolean[];
//   opaque: boolean[];
//   getBlockFaceMaterial: (id: BlockId, face: int) => MaterialId;
//   getMaterialData: (id: MaterialId) => Material;
//   renderer: Renderer;
//
//   constructor(registry: Registry, renderer: Renderer) {
//     this.solid = registry._solid;
//     this.opaque = registry._opaque;
//     this.getBlockFaceMaterial = registry.getBlockFaceMaterial.bind(registry);
//     this.getMaterialData = registry.getMaterialData.bind(registry);
//     this.renderer = renderer;
//   }
//
//   mesh(voxels: Tensor3): Mesh | null {
//     const data = this.computeGeometryData(voxels);
//     const numQuads = data.numQuads;
//     if (data.numQuads === 0) return null;
//
//     const geo = {
//       positions : new Float32Array(numQuads * 12),
//       normals   : new Float32Array(numQuads * 12),
//       indices   : new   Uint32Array(numQuads * 6),
//       colors    : new Float32Array(numQuads * 16),
//       uvws      : new Float32Array(numQuads * 12),
//     };
//
//     this.copyFloats(geo.positions, data.positions);
//     this.copyFloats(geo.normals,   data.normals);
//     this.copyInt32s(geo.indices,   data.indices);
//     this.copyFloats(geo.colors,    data.colors);
//     this.copyFloats(geo.uvws,      data.uvws);
//
//     return this.renderer.addFixedMesh(geo);
//   }
//
//   private copyInt32s(dst: Uint32Array, src: number[]) {
//     for (let i = 0; i < dst.length; i++) dst[i] = src[i];
//   }
//
//   private copyFloats(dst: Float32Array, src: number[]) {
//     for (let i = 0; i < dst.length; i++) dst[i] = src[i];
//   }
//
//   private addQuad(geo: GeometryData, d: int, w: int, h: int, mask: int,
//                   pos: Vec3, du: Vec3, dv: Vec3, normal: Vec3) {
//     const {numQuads, positions, normals, indices, colors, uvws} = geo;
//     geo.numQuads++;
//
//     const positions_offset = numQuads * 12;
//     const indices_offset   = numQuads * 6;
//     const colors_offset    = numQuads * 16;
//     const base_index       = numQuads * 4;
//
//     if (positions.length < positions_offset + 12) {
//       for (let i = 0; i < 12; i++) positions.push(0);
//       for (let i = 0; i < 12; i++) normals.push(0);
//       for (let i = 0; i < 6; i++)  indices.push(0);
//       for (let i = 0; i < 16; i++) colors.push(0);
//       for (let i = 0; i < 12; i++)  uvws.push(0);
//     }
//
//     for (let i = 0; i < 3; i++) {
//       positions[positions_offset + i + 0] = pos[i];
//       positions[positions_offset + i + 3] = pos[i] + du[i];
//       positions[positions_offset + i + 6] = pos[i] + du[i] + dv[i];
//       positions[positions_offset + i + 9] = pos[i] + dv[i];
//
//       const x = normal[i];
//       normals[positions_offset + i + 0] = x;
//       normals[positions_offset + i + 3] = x;
//       normals[positions_offset + i + 6] = x;
//       normals[positions_offset + i + 9] = x;
//     }
//
//     const triangleHint = this.getTriangleHint(mask);
//     const offsets = mask > 0
//       ? (triangleHint ? kIndexOffsets.C : kIndexOffsets.D)
//       : (triangleHint ? kIndexOffsets.A : kIndexOffsets.B);
//     for (let i = 0; i < 6; i++) {
//       indices[indices_offset + i] = base_index + offsets[i];
//     }
//
//     const id = Math.abs(mask >> 8) as MaterialId;
//     const material = this.getMaterialData(id);
//     let textureIndex = material.textureIndex;
//     if (textureIndex === 0 && material.texture) {
//       textureIndex = this.renderer.atlas.addImage(material.texture);
//       material.textureIndex = textureIndex;
//     }
//
//     const color = material.color;
//     for (let i = 0; i < 4; i++) {
//       const ao = 1 - 0.3 * (mask >> (2 * i) & 3);
//       colors[colors_offset + 4 * i + 0] = color[0] * ao;
//       colors[colors_offset + 4 * i + 1] = color[1] * ao;
//       colors[colors_offset + 4 * i + 2] = color[2] * ao;
//       colors[colors_offset + 4 * i + 3] = color[3];
//     }
//
//     const dir = Math.sign(mask);
//     for (let i = 0; i < 12; i++) uvws[positions_offset + i] = 0;
//     if (d === 2) {
//       uvws[positions_offset + 1] = uvws[positions_offset + 4] = h;
//       uvws[positions_offset + 3] = uvws[positions_offset + 6] = -dir * w;
//     } else {
//       uvws[positions_offset + 1] = uvws[positions_offset + 10] = w;
//       uvws[positions_offset + 6] = uvws[positions_offset + 9] = dir * h;
//     }
//     for (let i = 0; i < 4; i++) {
//       uvws[positions_offset + i * 3 + 2] = textureIndex;
//     }
//   }
//
//   private getTriangleHint(mask: int): boolean {
//     const a00 = (mask >> 0) & 3;
//     const a10 = (mask >> 2) & 3;
//     const a11 = (mask >> 4) & 3;
//     const a01 = (mask >> 6) & 3;
//     if (a00 === a11) return (a10 === a01) ? a10 === 3 : true;
//     return (a10 === a01) ? false : (a00 + a11 > a10 + a01);
//   }
