---
description: 'connect and serve: smaltml over any message port, such as a worker of your own.'
---

# connect

<!-- docs:skip -->
```ts
function connect(port: Port): MlWorker;
function serve(port: Port): void;
interface Port {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
  start?(): void;
}
```

`connect` talks to smaltml served on a port; `serve` answers on one. `startWorker` is the two with smaltml's own worker
script. Use them to serve smaltml from a worker of your own, beside your other work. `connect` is in
`@smaltworks/smaltml/worker`; `serve` too, for the worker's side.

## Parameters

- `port`: a `Worker`, a worker's `self`, or a `MessagePort`.

## Returns

`connect` returns an `MlWorker` ([`startWorker`](./start-worker)). `serve` returns nothing: it answers requests until
the port closes.

## Errors

As [`startWorker`](./start-worker).

## Examples

In your worker:

<!-- docs:skip -->
```ts
import { serve } from '@smaltworks/smaltml/worker';

serve(self);
```

On the page:

<!-- docs:skip -->
```ts
import { connect } from '@smaltworks/smaltml/worker';

const ml = connect(new Worker(new URL('./my-worker.js', import.meta.url), { type: 'module' }));
```

## See also

[`startWorker`](./start-worker)
