#include <napi.h>

#include <locale>
#include <sstream>

namespace {

class CommaDecimal : public std::numpunct<char> {
protected:
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};

struct RestoreLocale {
  std::locale previous;
  ~RestoreLocale() { std::locale::global(previous); }
};

Napi::Value WithCommaLocale(const Napi::CallbackInfo &info) {
  const std::locale locale(std::locale::classic(), new CommaDecimal);
  const RestoreLocale restore{std::locale::global(locale)};
  std::ostringstream number;
  number << 1234.5;
  const auto result =
      info[0].As<Napi::Function>().Call({Napi::String::New(info.Env(), number.str())});
  if (std::locale() != locale) {
    Napi::Error::New(info.Env(), "Preview formatting changed the global C++ locale")
        .ThrowAsJavaScriptException();
  }
  return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("withCommaLocale", Napi::Function::New(env, WithCommaLocale));
  return exports;
}

} // namespace

NODE_API_MODULE(test_locale, Init)
