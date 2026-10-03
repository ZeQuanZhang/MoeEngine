#!/bin/zsh
set -euo pipefail

root="${0:A:h:h}/vendor/local-dependencies"
mkdir -p "$root"

fetch_dependency() {
  local name="$1"
  local repository="$2"
  local revision="$3"
  local version="$4"
  local destination="$root/$name"
  local archive="/private/tmp/mference-$name-$revision.tar.gz"

  if [[ -e "$destination" ]]; then
    print -u2 "dependency destination already exists: $destination"
    return 1
  fi
  mkdir -p "$destination"
  curl -q --noproxy '*' --proxy '' --fail --location \
    --retry 4 --retry-all-errors --connect-timeout 20 --max-time 300 \
    "https://codeload.github.com/$repository/tar.gz/$revision" \
    --output "$archive"
  tar -xzf "$archive" --strip-components=1 -C "$destination"
  git -C "$destination" init -q
  git -C "$destination" add -A
  git -C "$destination" \
    -c user.name=MferenceBuild -c user.email=mference-build@localhost \
    commit -q -m "Local source archive $revision"
  git -C "$destination" tag "$version"
  print "prepared $name $version"
}

fetch_dependency eventsource mattt/EventSource \
  a3a85a85214caf642abaa96ae664e4c772a59f6e 1.4.1
fetch_dependency swift-asn1 apple/swift-asn1 \
  eb50cbd14606a9161cbc5d452f18797c90ef0bab 1.7.0
fetch_dependency swift-atomics apple/swift-atomics \
  b601256eab081c0f92f059e12818ac1d4f178ff7 1.3.0
fetch_dependency swift-collections apple/swift-collections \
  fea17c02d767f46b23070fdfdacc28a03a39232a 1.5.1
fetch_dependency swift-crypto apple/swift-crypto \
  1b6b2e274e85105bfa155183145a1dcfd63331f1 4.5.0
fetch_dependency swift-huggingface huggingface/swift-huggingface \
  b721959445b617d0bf03910b2b4aced345fd93bf 0.9.0
fetch_dependency swift-jinja huggingface/swift-jinja \
  0b67ecb79139f6addef8699eff3622808aa6c7dc 2.3.6
fetch_dependency swift-nio apple/swift-nio \
  f71c8d2a5e74a2c6d11a0fbe324774b5d6084237 2.99.0
fetch_dependency swift-system apple/swift-system \
  7c6ad0fc39d0763e0b699210e4124afd5041c5df 1.6.4
fetch_dependency swift-transformers huggingface/swift-transformers \
  2fa33e1f5e7131a7fc64c28e6d161dcec0d24820 1.3.3
fetch_dependency yyjson ibireme/yyjson \
  8b4a38dc994a110abaec8a400615567bd996105f 0.12.0
