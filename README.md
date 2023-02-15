# WAVE: WebAssembly Voxel Engine

This engine started out as a rewrite of [noa-engine](https://github.com/fenomas/noa).
It is now implemented in WebAssembly for a significant speedup. It also has
new features, including out-of-the-box support for:
  - Rendering faraway terrain with a level-of-detail system
  - Smooth, dynamic lighting based on cellular automata

## [Check out the demo!](https://www.skishore.me/voxels)

----

## Why rewrite?

Well, I wanted to write a 3D voxel game... Voxels for 3D games are like ASCII
graphics for a roguelike: by sacrificing realistic graphics, you can instead
build interlocking systems with emergent behavior.

I tried out some existing JavaScript voxel engines. They worked...alright.
noa-engine was the best of them - a clean implementation, a good feature set.
However, my personal laptop is an old MacBook with integrated graphics, and
pushing the draw distance even a bit causes it to skip frames.

So I started with noa-engine and rewrote it bit-by-bit. Soon I threw out the
[Bablyon.js](https://www.babylonjs.com/) renderer in favor of custom shaders
optimized for voxel terrain meshes. Now the basic engine works, and I'm going
to write the game, but I want to share the engine it to save other people the
trouble. Here's a few things that I like about it:

1. It has a code style that I prefer.
2. It handles level-of-detail and basic lighting out of the box.
3. It has several important algorithmic optimizations built in.

## Code style

Code style is very personal. Here are my opinions on this front:

- The source is in TypeScript. I find dynamically-typed code unreadable.

- I don't like dependencies, and I don't like splitting up code into one
  function per file. I really don't like NPM. It seems to generate an absurdly
  massive `node_module` directory, and without auditing what it's pulled in,
  I don't trust it. That said, I need TypeScript, so I vendored it in.

If this code isn't an NPM module, then how would you use it? Simple: check it
out and start modifying the code. You can make a whole game just by modifying
`main.js` and `worldgen.js`. Or, if you have alternative ideas about how to
organize your project, you can copy the bits you need into it. There's not too
much code here. It has a permissive license.

## Additional features

The biggest new feature is dynamic lighting. I haven't had time to write up
how the lighting works yet.

Level-of-detail, or LOD, is the basic mechanism by which games render a world
that seems to disappear into the horizon. The idea is to represent the world's
geometry at a coarser level-of-detail for terrain that is further away.

Level-of-detail is also annoying to implement efficiently for a voxel world.

Here it is, working out of the box. Like most voxel engines, this engine
divides up the world into chunks - in this case, 16 x 256 x 16 chunks. As in
Minecraft, the longer dimension here, (the y-dimension) is vertical, and the
x- and z-dimensions are horizontal. To load data for a chunk, the worldgen
callback must produce a "Column". This data structure represents the blocks at
a given (x, z) location in a
[run-length-encoded](https://en.wikipedia.org/wiki/Run-length_encoding) list.
Implementing this callback is sufficient to generate both full-resolution
"active" terrain, and an LOD heightmap for distant terrain.

(If you need to, you can implement separate callbacks for "active" chunks and
for the LOD frontier, so you don't need to place decorations like trees, carve
caves, etc. for faraway terrain. Decorations also don't need to fit into the
run-length encoded list - they can be specified separately.)

Another feature this engine handles is a basic lighting system. (noa-engine
has [ambient occlusion](https://en.wikipedia.org/wiki/Ambient_occlusion) -
this lighting is in addition to that.) Light falls vertically, i.e. in the
direction (0, -1, 0), so any block underneath the highest solid block in a
column is shaded. You probably want something more like Minecraft's lighting,
based on cellular automata. I do, too. It's coming.

## Algorithmic optimizations

I've put a lot of effort into optimizing this code. It runs smoothly on my
old MacBook, with over 10ms of time left over per frame for actual game logic.

One class of optimizations came from rewriting the renderer and shaders.
My implementation only targets WebGL2. Because WebGL2 supports
[2D texture arrays](https://github.com/WebGLSamples/WebGL2Samples/blob/master/samples/texture_2d_array.html),
it can make a single draw call to draw a chunk of voxels, regardless of the
number of distinct block types in that chunk. Doing that was a huge rendering
performance win. I also wrote a VoxelShader that uses the fact that all voxel
geometry is composed of axis-aligned quads. The shader reads these quads from
a compressed format and computes position, normal, etc. attributes on the fly.
It cut down geometry size from ~600Mb to ~120Mb!

This code also optimizes the CPU, i.e. JavaScript, side of the engine -
doing worldgen, constructing [greedy voxel meshes](https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/),
and maintaining active chunks and the LOD frontier. The key optimizations here
start with the data structure - the run-length encoded column. Simply using
a data structure like this makes worldgen much faster. It essentially turns a
problem that scales cubically - filling in blocks in a 16 x 256 x 16 chunk -
into one that scales quadratically.

Note that we do, eventually, produce the 3D tensor representation of a chunk.
However, we do so in the engine, by calling a TypedArray's "fill" method for
each run.  This method is native code that inflates the run-length encoding
much faster than pure JS. It's not a performance bottleneck.

### The equi-levels optimization

We can use run-length encoding to efficiently compute "equi-levels".
This algorithm yielded a massive speedup (5-10x) for greedy meshing.

An equi-level is a y-value in a chunk such that every block at that y-value
is the same. Chunks are (16, 16) in the (x, z) direction, so at an equi-level,
256 blocks are the same. It's possible to compute equi-levels in time that
scales with the number of runs, not the total height:

1. First, note that a level is an equi-level iff every column in the chunk
   has the same block at that level as some reference column. We arbitrarily
   choose the first column as the reference column.

2. We allocate a "changes" array, with an entry for each y-value (256 total).
   All the change entries start at 0.

3. After we've loaded each column, we compare it to the reference column using
   a two-finger algorithm, like a mergesort merge. At a y-value where the two
   columns transition from matching to unmatched, we increment changes[y]. At
   a y-value where they transition back to matched, we decrement changes[y].
   We only need to consider run endpoints here, so this step is fast.

4. Now, if we let "changesSum" be a cumulative sum array of the changes array,
   then changesSum[y] is equal to the number of columns at index y that have
   a different block from the reference column. A y-level is an equi-level iff
   changesSum[y] is 0.

So we can compute equi-levels quickly. In greedy meshing, if two consecutive
y-values are both equi-levels, and are either a) both opaque blocks, or b)
the same block, then we know that we don't need to produce any geometry for
those levels! Since voxel terrain tends to vary smoothly in the x- and z-
directions, this optimization skips meshing for 80-95% of columns!

Equi-levels interact well with decorations. When we decorate a cell, instead
of breaking the run it's a part of, we can just unset the equi-level flag for
that y-value. If the decoration is opaque (like a Minecraft ore), and we track
"opaque equi-levels" in addition to block-type equi-levels, we don't even need
to unset the flag.

----

# Summary

Voxels are a fun tool to play with.  Writing a performant voxel engine is...
well, it's also fun, but it's a distraction, if you want to finish a game.

If you know JavaScript, and you want to write a 3D game, learn from my
mistakes. Don't write your own engine. Use noa-engine, or use this one.

Getting started: `worldgen.ts` is a good place to try out some basic changes.
You can add gameplay by subscribing to more Input types in `engine.ts`, then
creating entities in the game world in `main.ts`.

Licensing notes: the MIT license applies to code in the `src` directory.
Code in the `lib` directory is in the public domain; see the comment at the
top of each file there for attribution. I do not have any rights to the use
of images in the `images` directory; they come from vanilla Minecraft,
Rhodox's Painterly Pack, or from Pokemon.
