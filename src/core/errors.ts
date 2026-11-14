/** The codes smaltml's errors carry. Each has an entry in the reference (docs/reference/errors.md). */
export type ErrorCode =
  /** The data's shape doesn't fit: rows that differ in length, `X` and `y` of different lengths, or a model fitted on another number of columns. */
  | 'ML_SHAPE'
  /** A value is missing (`NaN`) or infinite where the model can't take one. */
  | 'ML_VALUE'
  /** A parameter is out of range or of the wrong kind. */
  | 'ML_PARAM'
  /** `predict`, `transform` or `save` was called before `fit`. */
  | 'ML_NOT_FITTED'
  /** The data doesn't determine a single answer, such as linear regression on columns that repeat each other. */
  | 'ML_SINGULAR'
  /** A saved model that this version of smaltml can't read. */
  | 'ML_FORMAT'
  /** Labels the model doesn't support, such as one class only. */
  | 'ML_LABELS';

/** An error from smaltml, with a code that says what went wrong. */
export class SmaltmlError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = 'SmaltmlError';
    this.code = code;
  }
}

export function fail(code: ErrorCode, message: string): never {
  throw new SmaltmlError(code, message);
}
