open System

// ----------- Load data --------------

let datafolder = @"../../Dati"

// merge convolution files
//let csvfiles = System.IO.Directory.EnumerateFiles(datafolder, "FeatureExtractionResults.csv")
let csvfiles = System.IO.Directory.EnumerateFiles(datafolder, "FeatureExtractionResultsNoBranch.csv")
let filename = csvfiles |> Seq.head
let fieldNames =
    let lines = System.IO.File.ReadAllLines(filename)
    let header = lines |> Seq.head
    header.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)

let handleFile filename =
    let lines = System.IO.File.ReadAllLines(filename)
    let header = lines |> Seq.head
    let fields = header.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
    let handleLine (line:string) =
        let cells = line.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
        cells |> Array.map (fun f -> System.Double.Parse(f))
    lines |> Seq.skip 1 |> Seq.map handleLine

let allfields = csvfiles |> Seq.map handleFile |> Seq.concat |> Seq.toArray

fieldNames |> Array.iteri (fun i v -> System.Console.WriteLine(@"{0} {1}", i, v))

let getColumn i = allfields |> Array.map (fun ary -> ary.[i])

let inline sqr x = x * x
let normColumn col:float[] = 
    let avg = col |> Array.average
    let stddev = col |> Array.map ((-) avg) |> Array.map sqr |> Array.average |> sqrt
    //col |> Array.map (fun v -> (v - avg) / stddev)
    col

let alg = getColumn 25 

let discreteGPUTime = getColumn 22
let integratedGPUTime = getColumn 23
let CPUTime = getColumn 24 

let workSize = allfields |> Array.map (fun ary -> ary.[16] * ary.[17] * ary.[18])

let dataTrasfFromHost = getColumn 13
let dataTrasfFromDevice = getColumn 14

let readAccessGlobal = getColumn 1 
let writeAccessGlobal = getColumn 2 
let readAccessLocal = getColumn 3 
let writeAccessLocal = getColumn 4 
let readAccessConstant = getColumn 5 

let mulArrays (a:float[]) (b:float[]) = Array.map2 (fun v w -> v*w) a b 

let fixWithWorkSize a = mulArrays workSize a 

let arithmetic = fixWithWorkSize (getColumn 6)
let logic = fixWithWorkSize (getColumn 7) 
let bitwise = fixWithWorkSize (getColumn 8)
let comparison = fixWithWorkSize (getColumn 9)
let pow = fixWithWorkSize (getColumn 10)
let sqrt_op = fixWithWorkSize (getColumn 11)
let hypot = fixWithWorkSize (getColumn 12)

let sumArrays (a:float[]) (b:float[]) = Array.map2 (fun v w -> v+w) a b 
let simple_op =  (sumArrays logic bitwise)  |> sumArrays comparison
let complex_op = sumArrays arithmetic pow |> sumArrays sqrt_op |> sumArrays hypot 

//let readAccessGlobal = getColumn 1
//let writeAccessGlobal = getColumn 2
//let readAccessLocal = getColumn 3
//let writeAccessLocal = getColumn 4
//let readAccessConstant = getColumn 5

let divArrays (a:float[]) (b:float[]) = Array.map2 (fun v w -> v/w) a b 

let allMem = sumArrays readAccessGlobal writeAccessGlobal |> sumArrays readAccessLocal |> sumArrays writeAccessLocal |> sumArrays readAccessConstant
let memTot = mulArrays workSize allMem
let derived = divArrays (sumArrays (dataTrasfFromHost) (dataTrasfFromDevice)) memTot

#r @"MathNet.Numerics.dll"
#r @"MathNet.Numerics.IO.dll"
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra.Double

//let target = 2

//let basisSize = 2

let rnd = new System.Random()

// ------------- CON ANDREA ----------------
// M matrice misurazioni
// x surrogato
// b misure target program
// t target measure
// T misure target dei programmi benchmark
// t = T x
// M x = b
// x = M^{-1} b
// M = U S V*
// pseudoinversa di M = M* = U* S* V
// t = T x = T U* S* V b

let truncatedpseudoinverse n (M:LinearAlgebra.Generic.Matrix<float>) =
  let D = M.Svd(true)
  let W = D.W()
  let s = D.S()
  for i = 0 to min n (s.Count-1) do
    s.[i] <- if s.[i] = 0.0 then 0.0 else 1.0 / s.[i]
  W.SetDiagonal(s)
  (D.U() * W * D.VT()).Transpose()

let pseudoinverse (M:LinearAlgebra.Generic.Matrix<float>) =
  let D = M.Svd(true)
  let W = D.W()
  let s = D.S()
  let tolerance = Precision.EpsilonOf(2.0) * float(Math.Max(M.RowCount, M.ColumnCount)) * W.[0, 0]
  for i = 0 to s.Count-1 do
    s.[i] <- if s.[i] <= tolerance then 0.0 else 1.0 / s.[i]
  W.SetDiagonal(s)
  (D.U() * W * D.VT()).Transpose()

let inline cons a b =
  (a,b)

let inline sqr x =
  x * x

let isfinite x =
  not (Double.IsNaN(x) || Double.IsInfinity(x))

let inline print x =
  printf "%A" x

let inline printn x =
  printfn "%A" x

let inline csv s =
  let mutable i = 0
  for v in s do
    printfn "%A, %A" i v
    i <- i + 1

// -------------------------------------- CHECK MODEL ---------------------------------------

let distr = MathNet.Numerics.Distributions.ContinuousUniform()


let addNoise amount (M:LinearAlgebra.Generic.Matrix<float>) =
  let distr0 = MathNet.Numerics.Distributions.ContinuousUniform(-amount, amount)
  M + DenseMatrix.CreateRandom(M.RowCount, M.ColumnCount, distr0)

let multNoise amount (M:LinearAlgebra.Generic.Matrix<float>) =
  let distr0 = MathNet.Numerics.Distributions.ContinuousUniform(1.0 - amount, 1.0 + amount)
  M.PointwiseMultiply(DenseMatrix.CreateRandom(M.RowCount, M.ColumnCount, distr0))  

let noiseAmount = 0.1
let prepare = multNoise noiseAmount

//  ----- independent programs ----
let benchmarks = 4
let resources = 4
let A = DenseMatrix.CreateRandom(resources, benchmarks, distr)
let a = DenseMatrix.CreateRandom(100, benchmarks, distr)
let x =
  if true then
    DenseMatrix.CreateRandom(benchmarks, 1, distr)
  else
    DenseMatrix.Create(benchmarks, 1, (fun i j -> if i = 3 then 1.0 else 0.0))
let b = A * x

// add some noise
let A1 = prepare A
let a1 = prepare a
let b1 = prepare b

let inv = pseudoinverse A1
let predict = a1 * inv
let estimate = predict * b1
let exact = a * x

(*
printfn "%A\n%A" x (inv * b1)
printfn "%A" (predict)
*)

//estimate.PointwiseDivide(exact).Column(0) |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> csv

let test () =
  let preprocess (x:_[]) = [| x.[4]; x.[10] ; x.[13] ; x.[20] ; x.[30] |] |> Array.toSeq
  //let preprocess (x:_[]) = Seq.take 14 x
  let preprocessAll x = Seq.map preprocess x
  let getTarget (x:_[]) = [| x.[8]; x.[12] ; x.[18] ; x.[23] ; x.[32] |] |> Array.toSeq
  let getTargetAll x = Seq.map getTarget x
  let references = [| Array.create (dataTrasfFromHost.Length) 1.0 ; dataTrasfFromHost ; simple_op ; complex_op |]
  let hidden = [| discreteGPUTime ; integratedGPUTime ; CPUTime |]
  let numRef = references.Length
  let numHid = hidden.Length
  let p = preprocess references.[0] |> Seq.length
  let targets = getTarget references.[0] |> Seq.length
  let A = DenseMatrix.OfRows(numRef, p, preprocessAll references)
  let a = DenseMatrix.OfRows(numHid, p, preprocessAll hidden)
  let b = DenseMatrix.OfRows(numRef, targets, getTargetAll references)
  let exact = DenseMatrix.OfRows(numHid, targets, getTargetAll hidden)
  A |> printn
  a |> printn
  b |> printn
  let inv = pseudoinverse A
  let predict = a * inv
  let estimate = predict * b
  predict |> printn
  exact |> printn
  estimate |> printn



test()

#if TODO


let inv = pseudoinverse A1
let predict = a1 * inv
let estimate = predict * b1
let exact = a * x
estimate.PointwiseDivide(exact).Column(0) |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> Seq.toArray |> Array.mapi cons |> Chart.Line




// real data
(*
let publicData = DenseMatrix.OfRows(4,arithmetic.Length, [| dataTrasfFromDevice |> normColumn; 
                                                        readAccessGlobal |> normColumn; 
                                                        simple_op |> normColumn; 
                                                        complex_op  |> normColumn (*; derived *) |])
// hidden data
let hiddenData = DenseMatrix.OfRows(3,arithmetic.Length, [| discreteGPUTime |> normColumn; 
                                                        integratedGPUTime |> normColumn; 
                                                        CPUTime |> normColumn |])
*)




let check (benchmarkData:LinearAlgebra.Generic.Matrix<float>) =
  let cond = benchmarkData.ConditionNumber()


let preprocess (benchmarkData:LinearAlgebra.Generic.Matrix<float>) =
  let inv = pseudoinverse benchmarkData



dataTrasfFromDevice ; 
readAccessGlobal ; 
simple_op ; 
complex_op



// -------------------- NMF -------------------------


#r @"X:\Projects\EnergonWin8\Solvers\bin\Debug\Solvers.dll"
open Energon.Solvers

let NMFSolver = Energon.Solvers.predictUsingNonNegativeMatrixFactorization

let isfinite x =
      not (Double.IsNaN(x) || Double.IsInfinity(x))

type Result = {
 M: DenseMatrix;
 p1 : float;
 p2: float;
 p3: float;
 m1: float;
 m2: float;
 m3: float;
 cond: float;
}

let contains (ary: int array) (value: int) =
  ary |> Array.map (fun v -> v = value) |> Array.exists (id)

let randomizedTargetBasis targetAlg basisSize =
    // filter out cases with work size too little, as .net overhead is larger than actual completion time
    let lowerWorksize = 1024.0//*1024.0
    let filtered = workSize |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> v >= lowerWorksize) |> Array.map fst
    // take all the algorithm ids of each case, filter out cases with workload below lower limit
    let filteredIdxAlg = alg |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> contains filtered i)|> Seq.toArray
    // split cases into belonging to target algorithm or not
    let casesTargetAlg = filteredIdxAlg |> Array.filter (fun (i,v) -> v=targetAlg) |> Array.map fst
    let casesNoTargetAlg = filteredIdxAlg |> Array.filter (fun (i,v) -> v <> targetAlg || true) |> Array.map fst
    // pick a target case belonging to target algorithm
    let target = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.head
    // remove the target from the algorithms used in the basis (makes sense only if target alg is allowed in the base)
    let casesNoTargetAlgNoTarget = casesNoTargetAlg |> Array.filter (fun i -> i <> target)
    // pick a random base to build the model with
    let basis = casesNoTargetAlgNoTarget |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.take basisSize |> Seq.toArray
    target, basis

let normColumn (col:float array) =
  let m = Array.max col
  col |> Array.map (fun v -> v/m)

// zero centered
let normColumn col =
  let m = Array.average col
  let col0 = col |> Array.map (fun v -> v - m)
  let std = col0 |> Array.map sqr |> Array.average |> sqrt
  col0 |> Array.map ((*) (1.0 / std))

// not zero centered
let normColumn col =
  let m = Array.average col
  let col0 = col |> Array.map (fun v -> v - m)
  let std = col0 |> Array.map sqr |> Array.average |> sqrt
  col |> Array.map ((*) (1.0 / std))
  
let attempt target basis =
    (*let M_all = DenseMatrix.OfRows(5,arithmetic.Length, [| dataTrasfFromHost |> normColumn; // |> Array.map (fun v -> 1.0); 
                                                            dataTrasfFromDevice |> normColumn; 
                                                            readAccessGlobal |> normColumn; 
                                                            simple_op |> normColumn; 
                                                            complex_op  |> normColumn (*; derived *) |])
    *)
    let M_all = DenseMatrix.OfRows(4,arithmetic.Length, [| //dataTrasfFromHost ; // |> Array.map (fun v -> 1.0); 
                                                            dataTrasfFromDevice ; 
                                                            readAccessGlobal ; 
                                                            simple_op ; 
                                                            complex_op   (*; derived *) |])
    let b = DenseVector.OfEnumerable( M_all.Column(target) |> Seq.toArray |> normColumn )
    let M = DenseMatrix.OfColumnVectors(  M_all.ColumnEnumerator() |> Seq.filter(fun (i,v) -> contains basis i) |> Seq.map snd  |> Seq.toArray )
    // normalize rows
    let M = DenseMatrix.OfRows(M.RowCount, M.ColumnCount, M.RowEnumerator() |> Seq.map snd |> Seq.map (Seq.toArray) |> Seq.map normColumn |> Seq.map (fun ary -> (seq<float>) ary))

    let inv = pseudoinverse M
    let cond = M.ConditionNumber()

    let tryPredict (measures:float array) = 
        let a = measures |> Seq.mapi (fun i v -> i,v) |> Seq.filter (fun (i,v) -> contains basis i ) |> Seq.map snd |> Seq.toArray
        let measured = measures |> Seq.mapi (fun i v -> i,v) |> Seq.filter (fun (i,v) -> i = target ) |> Seq.map snd |> Seq.head
        let a = DenseMatrix.OfRows(1, a.Length, [| a |])
        //let prediction = (a * inv * b) |> Seq.head
        let prediction, _ = NMFSolver M b (DenseVector.OfVector( a.Row(0))) 2
        //printfn "%A * %A * %A = %A" a (inv*b) b prediction
        prediction, measured, (prediction - measured)/measured

    let p1,m1,e1 = tryPredict (discreteGPUTime |> normColumn )
    let p2,m2,e2 = tryPredict (integratedGPUTime  |> normColumn)
    let p3,m3,e3 = tryPredict (CPUTime |> normColumn)
    { M = M; p1 = p1; p2 = p2; p3 = p3; m1 = m1; m2 = m2; m3 = m3; cond = cond; }

let getM target basis =
    (*let M_all = DenseMatrix.OfRows(5,arithmetic.Length, [| dataTrasfFromHost |> normColumn; // |> Array.map (fun v -> 1.0); 
                                                            dataTrasfFromDevice |> normColumn; 
                                                            readAccessGlobal |> normColumn; 
                                                            simple_op |> normColumn; 
                                                            complex_op  |> normColumn (*; derived *) |])
    *)
    let M_all = DenseMatrix.OfRows(4,arithmetic.Length, [| //dataTrasfFromHost ; // |> Array.map (fun v -> 1.0); 
                                                            dataTrasfFromDevice |> normColumn; 
                                                            readAccessGlobal |> normColumn; 
                                                            simple_op |> normColumn; 
                                                            complex_op |> normColumn |])
    let b = DenseVector.OfEnumerable( M_all.Column(target) |> Seq.toArray |> normColumn )
    let M = DenseMatrix.OfColumnVectors(  M_all.ColumnEnumerator() |> Seq.filter(fun (i,v) -> contains basis i) |> Seq.map snd  |> Seq.toArray )
    // normalize rows
    let M = DenseMatrix.OfRows(M.RowCount, M.ColumnCount, M.RowEnumerator() |> Seq.map snd |> Seq.map (Seq.toArray) |> Seq.map normColumn |> Seq.map (fun ary -> (seq<float>) ary))
    M


// try using NMF
let trypredict algfamily basisSize =

    let rec doit () =
        try
            let target, basis = randomizedTargetBasis algfamily basisSize
            let M = getM target basis
            let publicData = DenseMatrix.OfRows(4,arithmetic.Length, [| dataTrasfFromDevice |> normColumn; 
                                                                    readAccessGlobal |> normColumn; 
                                                                    simple_op |> normColumn; 
                                                                    complex_op  |> normColumn (*; derived *) |])
            let hiddenData = DenseMatrix.OfRows(3,arithmetic.Length, [| discreteGPUTime |> normColumn; 
                                                                    integratedGPUTime |> normColumn; 
                                                                    CPUTime |> normColumn |])
            let a = DenseMatrix.OfColumnVectors(hiddenData.ColumnEnumerator() |> Seq.filter (fun (i,r) -> contains basis i) |> Seq.map snd |> Seq.toArray )
            let W,H,d = Energon.Solvers.NonNegativeMatrixFactorization M 3
            //let reconstructed = H * W
            //M - reconstructed
            let Wstar = pseudoinverse W
            let Hstar = pseudoinverse H
            let b = publicData.Column(target).ToColumnMatrix()
            let predicted = a * (Wstar * Hstar) * b
            let measured = DenseMatrix.OfColumnVectors(hiddenData.ColumnEnumerator() |> Seq.filter (fun (i,r) -> i = target) |> Seq.map snd |> Seq.toArray )
            let err = predicted.PointwiseDivide(measured).Column(0) |> Seq.toArray
            err
        with
        | _ -> doit ()
    let res = Array.init 1000 (fun _ -> doit ())
    res |> Array.concat

let res = trypredict 2.0 51
res |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> Seq.toArray |> Array.filter (fun v -> v < 2.0) |> Array.mapi cons |> Chart.Line


// H W = A
// h W = a
// H w = b
// h w = ?
// w = H* b
// h = a W*
// a W* H* b = ?


//  ----- independent programs ----
let benchmarks = 4
let resources = 4
let A = DenseMatrix.CreateRandom(resources, benchmarks, distr)
let a = DenseMatrix.CreateRandom(1000, benchmarks, distr)
let x = DenseMatrix.CreateRandom(benchmarks, 1, distr)
let b = A * x
// add some noise
let noiseAmount = 0.25
let A1 = A + DenseMatrix.CreateRandom(A.RowCount, A.ColumnCount, distr0) * noiseAmount //*0.0 // * 0.0 means no noise here
let a1 = a + DenseMatrix.CreateRandom(a.RowCount, a.ColumnCount, distr0) * noiseAmount //*0.0     // noise here
let b1 = b + DenseMatrix.CreateRandom(b.RowCount, b.ColumnCount, distr0) * noiseAmount //*0.0 // no noise here

let W,H,d = Energon.Solvers.NonNegativeMatrixFactorization A1 3

// a W* H* b = ?
let measure = a * x
let Wstar = pseudoinverse W
let Hstar = pseudoinverse H
let predicted = a1 * (Wstar * Hstar) * b1
let errors = predicted.PointwiseDivide(measure).Column(0) |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> Seq.toArray |> Array.mapi cons |> Chart.Line

let Aavg = A1.RowEnumerator() |> Seq.map (fun (i,v) -> v.Sum() / float(v.Count)) |> Seq.toArray
let aavg = a1.RowEnumerator() |> Seq.map (fun (i,v) -> v.Sum() / float(v.Count)) |> Seq.toArray

let A0 = DenseMatrix.OfRowVectors(A1.RowEnumerator() |> Seq.map (fun (i,v) -> v - Aavg.[i]) |> Seq.toArray)
let a0 = DenseMatrix.OfRowVectors(a1.RowEnumerator() |> Seq.map (fun (i,v) -> v - aavg.[i]) |> Seq.toArray)
let b0 = DenseMatrix.OfRowVectors(b1.RowEnumerator() |> Seq.map (fun (i,v) -> v - Aavg.[i]) |> Seq.toArray)


estimate.PointwiseDivide(exact).Column(0) |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> Seq.toArray |> Array.mapi cons |> Chart.Line




// cond
let doManyAttempts target basissize =
    let attempt2 targetAlg basisSize =
        let target, basis = randomizedTargetBasis targetAlg basisSize
        let r = attempt target basis
        r.cond
    let res = Array.init 100 (fun _ -> attempt2 target basissize)
    let resm1 = Array.map 
    let avg = res |> Array.average
    let prod = res |> Array.map abs |> Array.reduce (*)
    let geomAverage = Math.Pow(prod, 1.0 / float(res.Length))
    let stddev = res |> Array.map (fun v -> (v-avg)*(v-avg)) |> Array.average |> sqrt
    (avg, stddev, res.Length) //, stddev
    //geomAverage
let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )


// right prediction
// TODO DOES NOT MAKE SENSE IF I NORMALIZE COMPLETION TIMES INDIVIDUALLY
let doManyAttempts target basissize =
    let attempt2 targetAlg basisSize =
        let target, basis = randomizedTargetBasis targetAlg basisSize
        printf "%i %A\n" target basis
        let r = attempt target basis
        let predictions = [| 0,r.p3; 1,r.p1; 2,r.p2 |]
        let measures = [| 0,r.m3; 1,r.m1; 2,r.m2 |]
        let bestPredicted,_ = predictions |> Array.sortBy (fun (i,v) -> v) |> Seq.head
        let bestMeasured,_ = measures |> Array.sortBy (fun (i,v) -> v) |> Seq.head
        bestPredicted = bestMeasured

    let res = Array.init 100 (fun _ -> attempt2 target basissize) 
    let ok = res |> Array.filter (fun b -> b) |> Array.length
    float(ok)/float(res.Length)
let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (6 + 2*measures)) )

let doManyAttempts target basissize =
    let res = Array.init 100 (fun _ -> attempt target basissize) |> Array.filter isfinite
    let avg = res |> Array.average
    let prod = res |> Array.map abs |> Array.reduce (*)
    let geomAverage = Math.Pow(prod, 1.0 / float(res.Length))
    let stddev = res |> Array.map (fun v -> (v-avg)*(v-avg)) |> Array.average |> sqrt
    (avg, stddev, res.Length) //, stddev
    //geomAverage
let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )

Array.init 1000 (fun _ -> attempt 1.0 6) |> Array.filter isfinite |> Array.sort |> Array.mapi (fun i v -> (i,v)) |> Chart.Line

let doManyAttempts target basissize =
    let res = Array.init 100 (fun _ -> attempt target basissize) |> Array.map (fun (v1,v2,v3,c) -> [| v1; v2; v3 |]) |> Array.concat
    //let res = res |> Array.map sqr
    let avg = res |> Array.average
    let prod = res |> Array.map abs |> Array.reduce (*)
    let geomAverage = Math.Pow(prod, 1.0 / float(res.Length))
    let stddev = res |> Array.map (fun v -> (v-avg)*(v-avg)) |> Array.average |> sqrt
    avg, stddev
    //geomAverage

let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )

let r1,r2,r3 = Array.init 1000 (fun i -> attempt 3.0 20) |> Array.unzip3

let percentile (p:float) (data:float[]) =
    let sorted = data |> Array.map abs |> Array.sort
    let idx = int(float(sorted.Length) * p)
    sorted.[idx]

let percentiles data =
    percentile 0.25 data, percentile 0.5 data, percentile 0.75 data

let doCase alg size = 
    Array.init 1000 (fun i -> attempt alg size) |> Array.map (fun (a,b,c) -> [|a;b;c|]) |> Array.concat
let r = doCase 4.0 20

percentiles r




#endif

