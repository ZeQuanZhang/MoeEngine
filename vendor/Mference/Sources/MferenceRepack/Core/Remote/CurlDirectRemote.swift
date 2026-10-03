import Foundation

/// macOS direct-download transport. Some network extensions force CFNetwork
/// through a local proxy even when URLSession's proxy dictionary disables it.
/// `/usr/bin/curl --noproxy '*' --proxy ''` provides an explicit, observable
/// bypass while the repacker retains its strict range validation.
enum CurlDirectRemote {
    private struct Result {
        let data: Data
        let response: HTTPURLResponse
    }

    static func response(for request: URLRequest,
                         policy: RemoteDownloadSessionPolicy) throws
        -> (Data, URLResponse) {
        let result = try run(request: request,
                             outputPath: nil,
                             maximumRedirects: policy.maximumRedirects,
                             maximumBytes: requestedRangeLength(request),
                             allowedHostSuffix: policy.directAllowedHostSuffix,
                             timeoutSeconds: policy.requestTimeoutSeconds)
        return (result.data, result.response)
    }

    static func transfer(request: URLRequest,
                         targetPath: String,
                         expectation: RemoteRangeExpectation,
                         policy: RemoteDownloadSessionPolicy,
                         progress: @escaping @Sendable (UInt64) -> Void) throws
        -> RemoteRangeTransferResult {
        progress(0)
        do {
            guard try Posix.entryKind(targetPath) == .absent else {
                throw RepackError.installPathUnsafe(
                    path: targetPath,
                    detail: "direct-download target already exists")
            }
            let result = try run(request: request,
                                 outputPath: targetPath,
                                 maximumRedirects: policy.maximumRedirects,
                                 maximumBytes: expectation.length,
                                 allowedHostSuffix: policy.directAllowedHostSuffix,
                                 timeoutSeconds: policy.resourceTimeoutSeconds)
            try validate(result.response, expectation: expectation)
            let attributes = try FileManager.default.attributesOfItem(atPath: targetPath)
            let bytes = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
            guard bytes == expectation.length else {
                throw RepackError.remoteBodyTruncated(
                    path: expectation.filename,
                    expected: expectation.length,
                    actual: bytes)
            }
            _ = chmod(targetPath, 0o600)
            progress(bytes)
            return RemoteRangeTransferResult(path: targetPath, byteCount: bytes)
        } catch {
            try? FileManager.default.removeItem(atPath: targetPath)
            throw error
        }
    }

    private static func run(request: URLRequest,
                            outputPath: String?,
                            maximumRedirects: Int,
                            maximumBytes: UInt64?,
                            allowedHostSuffix: String?,
                            timeoutSeconds: TimeInterval) throws -> Result {
#if os(macOS)
        guard let url = request.url, url.scheme?.lowercased() == "https" else {
            throw RepackError.remoteProtocolInvalid(
                detail: "direct transport requires an HTTPS URL")
        }
        let temporaryRoot = NSTemporaryDirectory()
        let identifier = UUID().uuidString
        let headerPath = (temporaryRoot as NSString)
            .appendingPathComponent("mference-curl-\(identifier).headers")
        let bodyPath = outputPath ?? (temporaryRoot as NSString)
            .appendingPathComponent("mference-curl-\(identifier).body")
        defer {
            try? FileManager.default.removeItem(atPath: headerPath)
            if outputPath == nil {
                try? FileManager.default.removeItem(atPath: bodyPath)
            }
        }

        var arguments = [
            "-q",
            "--noproxy", "*",
            "--proxy", "",
            "--silent",
            "--show-error",
            "--location",
            "--max-redirs", String(maximumRedirects),
            "--proto", "=https",
            "--proto-redir", "=https",
            "--max-time", String(max(1, Int(timeoutSeconds))),
            "--dump-header", headerPath,
            "--output", bodyPath,
            "--write-out", "MFERENCE_CURL\t%{http_code}\t%{url_effective}\t%{size_download}",
        ]
        if request.httpMethod?.uppercased() == "HEAD" {
            arguments.append("--head")
        } else if let method = request.httpMethod, method.uppercased() != "GET" {
            arguments += ["--request", method]
        }
        for (name, value) in request.allHTTPHeaderFields ?? [:] {
            arguments += ["--header", "\(name): \(value)"]
        }
        arguments += ["--url", url.absoluteString]

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        process.arguments = arguments
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        var exceededLimit = false
        while process.isRunning {
            if let maximumBytes,
               let attributes = try? FileManager.default.attributesOfItem(atPath: bodyPath),
               let size = (attributes[.size] as? NSNumber)?.uint64Value,
               size > maximumBytes {
                exceededLimit = true
                process.terminate()
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        process.waitUntilExit()
        let output = String(
            data: stdout.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8) ?? ""
        let errorText = String(
            data: stderr.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8) ?? ""
        if exceededLimit {
            throw RepackError.remoteBodyExceeded(
                path: url.lastPathComponent,
                limit: maximumBytes ?? 0,
                attempted: ((try? FileManager.default.attributesOfItem(atPath: bodyPath)[.size])
                    as? NSNumber)?.uint64Value ?? 0)
        }
        guard process.terminationStatus == 0 else {
            if process.terminationStatus == 28 { throw URLError(.timedOut) }
            throw RepackError.remoteProtocolInvalid(
                detail: "direct curl failed (\(process.terminationStatus)): "
                    + errorText.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let fields = output.split(separator: "\t", omittingEmptySubsequences: false)
        guard fields.count == 4,
              fields[0] == "MFERENCE_CURL",
              let status = Int(fields[1]),
              let effectiveURL = URL(string: String(fields[2])) else {
            throw RepackError.remoteProtocolInvalid(
                detail: "could not parse direct curl result")
        }
        try validateEffectiveURL(effectiveURL, allowedHostSuffix: allowedHostSuffix)
        let headerData = try Data(contentsOf: URL(fileURLWithPath: headerPath))
        let headers = try finalHeaders(from: headerData)
        guard let response = HTTPURLResponse(
            url: effectiveURL,
            statusCode: status,
            httpVersion: nil,
            headerFields: headers) else {
            throw RepackError.remoteProtocolInvalid(
                detail: "could not construct direct HTTP response")
        }
        let data = outputPath == nil
            ? try Data(contentsOf: URL(fileURLWithPath: bodyPath))
            : Data()
        return Result(data: data, response: response)
#else
        throw RepackError.configurationInvalid(
            detail: "--direct curl transport is currently available on macOS only")
#endif
    }

    private static func requestedRangeLength(_ request: URLRequest) -> UInt64? {
        guard let value = request.value(forHTTPHeaderField: "Range"),
              value.hasPrefix("bytes="),
              let dash = value.firstIndex(of: "-") else { return nil }
        let startText = value[value.index(value.startIndex, offsetBy: 6)..<dash]
        let endText = value[value.index(after: dash)...]
        guard let start = UInt64(startText), let end = UInt64(endText), end >= start else {
            return nil
        }
        return end - start + 1
    }

    private static func finalHeaders(from data: Data) throws -> [String: String] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw RepackError.remoteProtocolInvalid(detail: "non-UTF8 direct response headers")
        }
        let lines = text.replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
        guard let start = lines.lastIndex(where: { $0.hasPrefix("HTTP/") }) else {
            throw RepackError.remoteProtocolInvalid(detail: "missing direct HTTP headers")
        }
        var headers: [String: String] = [:]
        for line in lines[(start + 1)...] {
            if line.isEmpty { break }
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces)
            let value = line[line.index(after: colon)...]
                .trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        return headers
    }

    private static func validateEffectiveURL(_ url: URL,
                                             allowedHostSuffix: String?) throws {
        guard url.scheme?.lowercased() == "https" else {
            throw RepackError.remoteRedirectRejected(
                url: redactRemoteURL(url),
                detail: "direct transfer left HTTPS")
        }
        guard let suffix = allowedHostSuffix?.lowercased(), !suffix.isEmpty else { return }
        let host = url.host?.lowercased() ?? ""
        guard host == suffix || host.hasSuffix(".\(suffix)") else {
            throw RepackError.remoteRedirectRejected(
                url: redactRemoteURL(url),
                detail: "direct transfer left the allowed \(suffix) domain")
        }
    }

    private static func validate(_ response: HTTPURLResponse,
                                 expectation: RemoteRangeExpectation) throws {
        guard response.statusCode == 206 || response.statusCode == 200 else {
            throw RepackError.remoteHTTPResponse(
                url: redactRemoteURL(response.url),
                status: response.statusCode,
                retryAfter: remoteHeader(response, "Retry-After"))
        }
        guard let contentLength = remoteHeader(response, "Content-Length"),
              UInt64(contentLength) == expectation.length else {
            throw RepackError.remoteProtocolInvalid(
                detail: "wrong Content-Length for \(expectation.filename)")
        }
        guard let range = remoteHeader(response, "Content-Range") else {
            throw RepackError.remoteProtocolInvalid(
                detail: "missing Content-Range for \(expectation.filename)")
        }
        let end = expectation.offset + expectation.length - 1
        let parts = range.dropFirst("bytes ".count).split(separator: "/", maxSplits: 1)
        let bounds = parts.first?.split(separator: "-", maxSplits: 1) ?? []
        guard range.hasPrefix("bytes "), parts.count == 2, bounds.count == 2,
              UInt64(bounds[0]) == expectation.offset,
              UInt64(bounds[1]) == end,
              UInt64(parts[1]) == expectation.totalSize else {
            throw RepackError.remoteProtocolInvalid(detail: "wrong Content-Range \(range)")
        }
        if let expected = expectation.xetHash {
            let actual = normalizedStrongETag(remoteHeader(response, "X-Linked-ETag"))
                ?? normalizedStrongETag(remoteHeader(response, "ETag"))
            guard actual == expected.lowercased() else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "final ranged validator differs for \(expectation.filename)")
            }
        }
    }
}
